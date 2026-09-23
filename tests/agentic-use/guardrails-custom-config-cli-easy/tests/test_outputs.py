# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that the agent created a custom guardrail configuration with keyword-based
input and output rails using a real LLM for content evaluation.

Checks:
- harbor-custom-config exists with correct description
- Config has both input and output rails configured
- Config uses the guardrails-llm model
- Config has prompts for self_check_input (fruit blocking) and self_check_output (bread blocking)
- Input rail blocks messages mentioning fruit
- Normal messages pass through both rails
- Output rail blocks responses about baking bread
- Agent performed the expected CRUD and inference operations
"""

import base64
import json
import os

import pytest
from nemo_helix_plugin.guardrail.client import GuardrailClient
from nemo_helix_plugin.guardrail.types import GuardrailCheckRequest, GuardrailConfig
from trace_reader import get_session

WORKSPACE = "default"
CONFIG_NAME = "harbor-custom-config"
CONFIG_ID = f"{WORKSPACE}/{CONFIG_NAME}"
MODEL = "default/guardrails-llm"


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
def client() -> GuardrailClient:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return GuardrailClient(base_url=nhx_base_url, workspace=WORKSPACE, auth=_make_unsigned_jwt())


@pytest.fixture
def config(client: GuardrailClient) -> GuardrailConfig:
    """Retrieve the agent-created guardrail config."""
    return client.get_guardrail_config(name=CONFIG_NAME).data()


# --- Config structure checks ---


def test_config_exists(config: GuardrailConfig) -> None:
    """Test that harbor-custom-config was created."""
    assert config.name == CONFIG_NAME, f"Expected config name '{CONFIG_NAME}', got '{config.name}'"
    print(f"Config exists: {config.name}")


def test_config_description_updated(config: GuardrailConfig) -> None:
    """Test that the config description was updated."""
    assert config.description == "Updated custom guardrail config", (
        f"Expected description 'Updated custom guardrail config', got '{config.description}'"
    )


def test_config_has_input_rails(config: GuardrailConfig) -> None:
    """Test that the config has input rails configured."""
    rails = config.data.get("rails") or {}
    input_flows = (rails.get("input") or {}).get("flows") or []
    assert any("self check input" in f for f in input_flows), (
        f"Expected 'self check input' in input rail flows, got {input_flows}"
    )
    print(f"Input rails configured: {input_flows}")


def test_config_has_output_rails(config: GuardrailConfig) -> None:
    """Test that the config has output rails configured."""
    rails = config.data.get("rails") or {}
    output_flows = (rails.get("output") or {}).get("flows") or []
    assert any("self check output" in f for f in output_flows), (
        f"Expected 'self check output' in output rail flows, got {output_flows}"
    )
    print(f"Output rails configured: {output_flows}")


def test_config_uses_guardrails_model(config: GuardrailConfig) -> None:
    """Test that the config uses the guardrails-llm model."""
    models = config.data.get("models") or []
    model_names = [m.get("model", "") for m in models]
    assert any("guardrails-llm" in name for name in model_names), (
        f"Expected a model containing 'guardrails-llm', got {model_names}"
    )
    print(f"Models configured: {model_names}")


def test_config_has_input_prompt_about_fruit(config: GuardrailConfig) -> None:
    """Test that the self_check_input prompt checks for fruit mentions."""
    prompts = config.data.get("prompts") or []
    input_prompts = [p for p in prompts if "self_check_input" in p.get("task", "")]
    assert len(input_prompts) > 0, (
        f"Expected a prompt with task 'self_check_input', got tasks: {[p.get('task') for p in prompts]}"
    )
    content = (input_prompts[0].get("content") or "").lower()
    assert "fruit" in content, f"Expected self_check_input prompt to mention 'fruit', got: {content[:200]}"


def test_config_has_output_prompt_about_bread(config: GuardrailConfig) -> None:
    """Test that the self_check_output prompt checks for bread baking content."""
    prompts = config.data.get("prompts") or []
    output_prompts = [p for p in prompts if "self_check_output" in p.get("task", "")]
    assert len(output_prompts) > 0, (
        f"Expected a prompt with task 'self_check_output', got tasks: {[p.get('task') for p in prompts]}"
    )
    content = (output_prompts[0].get("content") or "").lower()
    assert "bread" in content or "baking" in content, (
        f"Expected self_check_output prompt to mention 'bread' or 'baking', got: {content[:200]}"
    )


# --- Functional inference checks ---


def test_input_rail_blocks_fruit_mention(client: GuardrailClient) -> None:
    """Test that a message mentioning fruit is blocked by the input rail.

    The self_check_input prompt tells the LLM to block messages mentioning fruit.
    A message about apples should trigger a 'Yes' response from the self-check,
    causing guardrails to mark the request blocked.
    """
    response = client.check_guardrail(
        body=GuardrailCheckRequest(
            model=MODEL,
            messages=[{"role": "user", "content": "Tell me about the health benefits of apples"}],
            guardrails={"config_id": CONFIG_ID},
            max_tokens=256,
            temperature=0,
        )
    ).data()
    assert response.status == "blocked", f"Message mentioning fruit should be blocked, got: {response.status}"
    print(f"Input rail correctly blocked fruit mention: {response.status}")


def test_normal_message_passes_through(client: GuardrailClient) -> None:
    """Test that a normal message (no fruit, no bread) passes through both rails.

    A geography question doesn't mention fruit (passes input rail) and the response
    won't be about bread baking (passes output rail).
    """
    response = client.check_guardrail(
        body=GuardrailCheckRequest(
            model=MODEL,
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            guardrails={"config_id": CONFIG_ID},
            max_tokens=256,
            temperature=0,
        )
    ).data()
    assert response.status == "success", f"Normal message should NOT be blocked, got: {response.status}"
    print(f"Normal message passed through: {response.status}")


def test_output_rail_blocks_bread_content(client: GuardrailClient) -> None:
    """Test that a response about baking bread is blocked by the output rail.

    The message doesn't mention fruit (passes input rail), but asking about bread
    baking will elicit a response about baking bread, which the output self-check
    should mark as blocked.
    """
    response = client.check_guardrail(
        body=GuardrailCheckRequest(
            model=MODEL,
            messages=[{"role": "user", "content": "Give me a step-by-step guide for baking sourdough bread"}],
            guardrails={"config_id": CONFIG_ID},
            max_tokens=256,
            temperature=0,
        )
    ).data()
    assert response.status == "blocked", f"Response about baking bread should be blocked, got: {response.status}"
    print(f"Output rail correctly blocked bread content: {response.status}")


# --- Trajectory check ---


def test_agent_performed_operations() -> None:
    """Verify the agent performed the expected CRUD and inference operations via trajectory."""
    session = get_session()
    commands = session.get_bash_commands()

    def has_command(*patterns: str) -> bool:
        return any(all(p in cmd for p in patterns) for cmd in commands)

    # Agent should have set up the inference provider
    assert has_command("secrets", "create", "nvidia-api-key") or has_command("secret", "create", "nvidia-api-key"), (
        f"Agent did not create the nvidia-api-key secret. Commands: {commands}"
    )

    # Agent should have listed or inspected existing configs
    assert has_command("guardrail", "configs"), f"Agent did not interact with guardrail configs. Commands: {commands}"

    # Agent should have created the custom config
    assert has_command("guardrail", "configs", "create", "harbor-custom-config"), (
        f"Agent did not create 'harbor-custom-config'. Commands: {commands}"
    )

    # Agent should have retrieved the config
    assert has_command("guardrail", "configs", "get", "harbor-custom-config"), (
        f"Agent did not retrieve 'harbor-custom-config'. Commands: {commands}"
    )

    # Agent should have updated the config
    assert has_command("guardrail", "configs", "update", "harbor-custom-config"), (
        f"Agent did not update 'harbor-custom-config'. Commands: {commands}"
    )

    # Agent should have used the migrated guardrail check endpoint
    made_inference = has_command("guardrail", "check")
    assert made_inference, f"Agent did not make a guardrail check call. Commands: {commands}"

    print(f"All trajectory checks passed. Total commands: {len(commands)}")

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that model provider registration operations were performed correctly.

Checks:
- Final state: harbor-provider-api-key secret exists
- Final state: harbor-test-provider no longer exists (was deleted)
- Final state: harbor-final-provider exists with correct host URL, description, and secret ref
- Trace: agent actually created harbor-test-provider (not just skipped it)
- Trace: agent actually deleted harbor-test-provider (not just never created it)
"""

import os

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.secrets.client import SecretsClient
from trace_reader import get_session

WORKSPACE = "default"


@pytest.fixture
def client() -> NeMoHelix:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return NeMoHelix(base_url=nhx_base_url, workspace=WORKSPACE)


# --- Final state checks ---


def test_api_key_secret_exists(client: NeMoHelix) -> None:
    """Test that the API key secret was created for provider registration."""
    secrets = client_from_platform(client, SecretsClient)
    response = secrets.get_secret(name="harbor-provider-api-key").data()
    assert response.name == "harbor-provider-api-key", (
        f"Expected secret name 'harbor-provider-api-key', got '{response.name}'"
    )


def test_harbor_test_provider_deleted(client: NeMoHelix) -> None:
    """Test that harbor-test-provider was deleted after initial registration."""
    response = client.inference.providers.list()
    provider_names = [p.name for p in response.data]
    assert "harbor-test-provider" not in provider_names, (
        f"Provider 'harbor-test-provider' should have been deleted but still exists! Found: {provider_names}"
    )


def test_harbor_final_provider_exists(client: NeMoHelix) -> None:
    """Test that harbor-final-provider was registered with correct settings."""
    response = client.inference.providers.retrieve(name="harbor-final-provider")
    assert response.name == "harbor-final-provider", (
        f"Expected provider name 'harbor-final-provider', got '{response.name}'"
    )
    assert response.host_url == "https://integrate.api.nvidia.com/v1", (
        f"Expected host URL 'https://integrate.api.nvidia.com/v1', got '{response.host_url}'"
    )
    assert response.description == "Final provider for verification", (
        f"Expected description 'Final provider for verification', got '{response.description}'"
    )


def test_harbor_final_provider_has_secret_ref(client: NeMoHelix) -> None:
    """Test that harbor-final-provider references the correct API key secret."""
    response = client.inference.providers.retrieve(name="harbor-final-provider")
    assert response.api_key_secret_name == "harbor-provider-api-key", (
        f"Expected api_key_secret_name 'harbor-provider-api-key', got '{response.api_key_secret_name}'"
    )


# --- Trace checks: verify intermediate steps actually occurred ---


def test_agent_created_test_provider() -> None:
    """Verify the agent actually ran a command to create harbor-test-provider."""
    session = get_session()
    commands = session.get_bash_commands()
    assert any("providers" in cmd and "create" in cmd and "harbor-test-provider" in cmd for cmd in commands), (
        f"Agent never created 'harbor-test-provider'. Commands run: {commands}"
    )


def test_agent_deleted_test_provider() -> None:
    """Verify the agent actually ran a command to delete harbor-test-provider."""
    session = get_session()
    commands = session.get_bash_commands()
    assert any("providers" in cmd and "delete" in cmd and "harbor-test-provider" in cmd for cmd in commands), (
        f"Agent never deleted 'harbor-test-provider'. Commands run: {commands}"
    )

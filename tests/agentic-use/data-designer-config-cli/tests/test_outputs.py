# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that the Data Designer model configuration was created correctly.

This test checks:
1. The secret dd-test-api-key was created
2. The model provider dd-test-provider was created with correct settings
"""

import os

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.secrets.client import SecretsClient


def test_secret_created() -> None:
    """Test that the dd-test-api-key secret was successfully created."""
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    client = NemoClient(base_url=nhx_base_url, workspace="default")

    # List secrets and check for our test secret
    secrets = SecretsClient.from_client(client)
    secret_names = [s.name for s in secrets.list_secrets().items()]

    assert "dd-test-api-key" in secret_names, f"Secret 'dd-test-api-key' was not created! Found secrets: {secret_names}"
    print("Test passed: dd-test-api-key secret was successfully created")


def test_provider_created() -> None:
    """Test that the dd-test-provider model provider was successfully created."""
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    client = NemoClient(base_url=nhx_base_url, workspace="default")

    # List providers and check for our test provider
    response = ModelsClient.from_client(client).list_providers()
    provider_names = [p.name for p in response.items()]

    assert "dd-test-provider" in provider_names, (
        f"Provider 'dd-test-provider' was not created! Found providers: {provider_names}"
    )
    print("Test passed: dd-test-provider model provider was successfully created")


def test_provider_configuration() -> None:
    """Test that the provider has the correct configuration."""
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    client = NemoClient(base_url=nhx_base_url, workspace="default")

    # Get the specific provider
    provider = ModelsClient.from_client(client).get_provider(name="dd-test-provider").data()

    # Verify host URL
    assert provider.host_url == "https://integrate.api.nvidia.com", (
        f"Provider host_url is incorrect! Expected 'https://integrate.api.nvidia.com', got '{provider.host_url}'"
    )

    # Verify API key secret reference
    assert provider.api_key_secret_name == "dd-test-api-key", (
        f"Provider api_key_secret_name is incorrect! Expected 'dd-test-api-key', got '{provider.api_key_secret_name}'"
    )

    print("Test passed: dd-test-provider has correct configuration")

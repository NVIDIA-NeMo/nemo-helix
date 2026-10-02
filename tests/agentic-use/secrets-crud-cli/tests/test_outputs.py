# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that secret CRUD operations were performed correctly.

Checks:
- harbor-test-secret was deleted (should not exist)
- harbor-final-secret exists with correct description
- Secret values are never exposed in API responses
"""

import os

import pytest
from nemo_helix_plugin.secrets.client import SecretsClient

WORKSPACE = "default"


@pytest.fixture
def client() -> SecretsClient:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return SecretsClient(base_url=nhx_base_url, workspace=WORKSPACE)


def test_harbor_test_secret_deleted(client: SecretsClient) -> None:
    """Test that harbor-test-secret was deleted after CRUD operations."""
    secret_names = [s.name for s in client.list_secrets().items()]
    assert "harbor-test-secret" not in secret_names, (
        f"Secret 'harbor-test-secret' should have been deleted but still exists! Found: {secret_names}"
    )


def test_harbor_final_secret_exists(client: SecretsClient) -> None:
    """Test that harbor-final-secret was created and has correct metadata."""
    response = client.get_secret(name="harbor-final-secret").data()
    assert response.name == "harbor-final-secret", f"Expected secret name 'harbor-final-secret', got '{response.name}'"
    assert response.description == "Final secret for verification", (
        f"Expected description 'Final secret for verification', got '{response.description}'"
    )


def test_secret_value_not_exposed(client: SecretsClient) -> None:
    """Test that secret values are not exposed in list/retrieve responses."""
    response = client.get_secret(name="harbor-final-secret").data()
    # HelixSecretResponse should only contain metadata fields, not the secret data.
    # The actual secret value is only available via client.access_secret().
    response_dict = response.model_dump()
    assert "data" not in response_dict or response_dict.get("data") is None, (
        "Secret value should not be exposed in retrieve response"
    )

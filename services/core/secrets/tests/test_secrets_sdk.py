# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the secrets service via the typed SecretsClient.

These tests verify:
- Secret CRUD operations (create, retrieve, list, update, delete)
- Secret access (value retrieval)
- Validation (empty data rejection)

Uses the create_test_client pattern for fast in-memory testing.
"""

import uuid

from nemo_helix_plugin.client.errors import NotFoundError, UnprocessableEntityError
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest, HelixSecretUpdateRequest
from nhx.common.entities import DEFAULT_WORKSPACE


def short_secret_name(prefix: str) -> str:
    """Generate a short secret name (max 32 chars total)."""
    suffix = uuid.uuid4().hex[:8]
    return f"{prefix[:22]}-{suffix}"


def test_create_secret(sdk: SecretsClient):
    secret_name = short_secret_name("testsecret")
    secret_value = "supersecret"
    secret = sdk.create_secret(
        workspace=DEFAULT_WORKSPACE, body=HelixSecretCreateRequest(name=secret_name, value=secret_value)
    ).data()
    assert secret.name == secret_name
    # Retrieve the secret
    secret_retrieved = sdk.get_secret(name=secret_name, workspace=DEFAULT_WORKSPACE).data()
    assert secret_retrieved.name == secret.name
    # Access the secret value
    secret_access_resp = sdk.access_secret(name=secret_name, workspace=DEFAULT_WORKSPACE).data()
    assert secret_access_resp.value == secret_value


def test_create_and_list_secrets(sdk: SecretsClient):
    secret_name_1 = short_secret_name("secret1")
    secret_name_2 = short_secret_name("secret2")
    secret_data = "somedata"
    sdk.create_secret(workspace=DEFAULT_WORKSPACE, body=HelixSecretCreateRequest(name=secret_name_1, value=secret_data))
    sdk.create_secret(workspace=DEFAULT_WORKSPACE, body=HelixSecretCreateRequest(name=secret_name_2, value=secret_data))
    # List secrets and verify both are present
    secret_names = [secret.name for secret in sdk.list_secrets(workspace=DEFAULT_WORKSPACE).items()]
    assert secret_name_1 in secret_names
    assert secret_name_2 in secret_names


def test_create_and_list_secrets_with_pagination(sdk: SecretsClient):
    """Test listing secrets with pagination across multiple pages."""
    num_secrets = 25
    secret_data = "paginationtest"

    # Create 25 secrets
    created_secret_names = []
    for i in range(num_secrets):
        secret_name = short_secret_name(f"page{i:02d}")
        sdk.create_secret(
            workspace=DEFAULT_WORKSPACE, body=HelixSecretCreateRequest(name=secret_name, value=secret_data)
        )
        created_secret_names.append(secret_name)

    # The paginated response iterates across all pages transparently.
    for secret in sdk.list_secrets(workspace=DEFAULT_WORKSPACE).items():
        if secret.name in created_secret_names:
            created_secret_names.remove(secret.name)
    assert len(created_secret_names) == 0, "Not all created secrets were found in the list"


def test_create_secret_with_empty_data(sdk: SecretsClient):
    secret_name = short_secret_name("emptydata")
    # An empty value is rejected client-side by the request model validator.
    try:
        HelixSecretCreateRequest(name=secret_name, value="")
        assert False, "Expected a validation error when creating a secret with empty data"
    except ValueError:
        pass


def test_create_secret_with_empty_data_server_side(sdk: SecretsClient):
    """If an empty value reaches the server, it responds 422."""
    secret_name = short_secret_name("emptydata")
    body = HelixSecretCreateRequest.model_construct(name=secret_name, value=_EmptySecret())
    try:
        sdk.create_secret(workspace=DEFAULT_WORKSPACE, body=body)
        assert False, "Expected a 422 when creating a secret with empty data"
    except UnprocessableEntityError as e:
        assert e.status_code == 422


class _EmptySecret:
    """Stand-in that serializes to an empty string to exercise server validation."""

    def get_secret_value(self) -> str:
        return ""


def test_create_and_delete_secret(sdk: SecretsClient):
    secret_name = short_secret_name("secret1")
    secret_value = "deletesecret"
    create_resp = sdk.create_secret(
        workspace=DEFAULT_WORKSPACE, body=HelixSecretCreateRequest(name=secret_name, value=secret_value)
    ).data()
    assert secret_name == create_resp.name
    sdk.delete_secret(name=secret_name, workspace=DEFAULT_WORKSPACE)
    try:
        sdk.get_secret(name=secret_name, workspace=DEFAULT_WORKSPACE)
        assert False, "Expected an error when retrieving a deleted secret"
    except NotFoundError as e:
        assert e.status_code == 404


def test_update_secret(sdk: SecretsClient):
    secret_name = short_secret_name("update")
    secret_value = "initialvalue"
    create_resp = sdk.create_secret(
        workspace=DEFAULT_WORKSPACE, body=HelixSecretCreateRequest(name=secret_name, value=secret_value)
    ).data()
    assert secret_name == create_resp.name
    assert create_resp.description is None
    # Update the secret's description
    updated_secret = sdk.update_secret(
        name=secret_name,
        workspace=DEFAULT_WORKSPACE,
        body=HelixSecretUpdateRequest(description="Updated description"),
    ).data()
    assert updated_secret.description == "Updated description"
    # Update description and value together
    updated_secret = sdk.update_secret(
        name=secret_name,
        workspace=DEFAULT_WORKSPACE,
        body=HelixSecretUpdateRequest(description="", value="newvalue"),
    ).data()
    assert updated_secret.description == ""
    assert updated_secret.name == secret_name
    # Access the updated secret value
    secret_access_resp = sdk.access_secret(name=secret_name, workspace=DEFAULT_WORKSPACE).data()
    assert secret_access_resp.value == "newvalue"


def test_rotate_encryption_keys(sdk: SecretsClient):
    """Test that secret rotation via the client preserves secret data.

    This test verifies:
    1. Secrets can be created with the client
    2. The rotate_encryption_keys admin endpoint can be called via the client
    3. After rotation, all secrets remain accessible with their original values
    """
    secrets_data = [
        (short_secret_name("rotate1"), "secret-value-1"),
        (short_secret_name("rotate2"), "secret-value-2"),
        (short_secret_name("rotate3"), "another-secret-value"),
    ]

    for secret_name, secret_value in secrets_data:
        sdk.create_secret(
            workspace=DEFAULT_WORKSPACE, body=HelixSecretCreateRequest(name=secret_name, value=secret_value)
        )

    for secret_name, expected_value in secrets_data:
        access_resp = sdk.access_secret(name=secret_name, workspace=DEFAULT_WORKSPACE).data()
        assert access_resp.value == expected_value

    # Rotate encryption keys
    sdk.rotate_encryption_keys()

    for secret_name, expected_value in secrets_data:
        access_resp = sdk.access_secret(name=secret_name, workspace=DEFAULT_WORKSPACE).data()
        assert access_resp.value == expected_value, f"Secret {secret_name} value changed after rotation"


def test_rotate_encryption_keys_idempotent(sdk: SecretsClient):
    """Test that calling rotate_encryption_keys multiple times is safe."""
    secret_name = short_secret_name("idempotent")
    secret_value = "idempotent-test-value"
    sdk.create_secret(workspace=DEFAULT_WORKSPACE, body=HelixSecretCreateRequest(name=secret_name, value=secret_value))

    for _ in range(3):
        sdk.rotate_encryption_keys()

    access_resp = sdk.access_secret(name=secret_name, workspace=DEFAULT_WORKSPACE).data()
    assert access_resp.value == secret_value

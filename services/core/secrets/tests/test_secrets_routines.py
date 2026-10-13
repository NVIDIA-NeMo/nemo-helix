# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for secret rotation routines."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from nhx.common.entities.client import ListResponse, PaginationInfo
from nhx.common.secrets.encryption import (
    SecretKeyEncryptor,
    envelope_decrypt,
    envelope_encrypt,
)
from nhx.core.secrets.api.v2.admin.routines import rotate_encryption_keys
from nhx.core.secrets.entities import HelixSecret


def make_secret(
    name: str,
    workspace: str,
    encryptor: SecretKeyEncryptor,
    secret_data: str,
) -> HelixSecret:
    """Create a HelixSecret with envelope encryption.

    Uses envelope encryption to encrypt the secret data with a randomly
    generated DEK, then encrypts the DEK with the provided KEK encryptor.

    Args:
        name: The secret name.
        workspace: The workspace name.
        encryptor: The KEK encryptor to use for encrypting the DEK.
        secret_data: The plaintext secret data to encrypt.

    Returns:
        A HelixSecret with encrypted data, encrypted DEK, and provider set.
    """
    encrypted_data, encrypted_dek, provider = envelope_encrypt(encryptor, secret_data)
    secret = HelixSecret(name=name, workspace=workspace, description="Test secret")
    secret._data = encrypted_data
    secret._encrypted_dek = encrypted_dek
    secret._secret_provider = provider
    return secret


def make_list_response(secrets: list[HelixSecret]) -> ListResponse[HelixSecret]:
    """Create a ListResponse with the given secrets."""
    return ListResponse(
        data=secrets,
        pagination=PaginationInfo(
            page=1,
            page_size=1000,
            current_page_size=len(secrets),
            total_pages=1,
            total_results=len(secrets),
        ),
    )


def make_paginated_list_responses(secrets: list[HelixSecret], page_size: int = 100) -> list[ListResponse[HelixSecret]]:
    """Create a list of paginated ListResponses for the given secrets.

    Args:
        secrets: All secrets to paginate.
        page_size: Number of secrets per page.

    Returns:
        List of ListResponse objects, one per page.
    """
    total_results = len(secrets)
    total_pages = (total_results + page_size - 1) // page_size if total_results > 0 else 1

    responses = []
    for page in range(1, total_pages + 1):
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_results)
        page_secrets = secrets[start_idx:end_idx]

        responses.append(
            ListResponse(
                data=page_secrets,
                pagination=PaginationInfo(
                    page=page,
                    page_size=page_size,
                    current_page_size=len(page_secrets),
                    total_pages=total_pages,
                    total_results=total_results,
                ),
            )
        )

    # Handle empty list case
    if not responses:
        responses.append(
            ListResponse(
                data=[],
                pagination=PaginationInfo(
                    page=1,
                    page_size=page_size,
                    current_page_size=0,
                    total_pages=1,
                    total_results=0,
                ),
            )
        )

    return responses


@pytest.mark.asyncio
async def test_rotate_encryption_keys_happy_path(mocker):
    """Test that all secrets encrypted with v1 are re-encrypted with v2.

    Given a list of secrets encrypted with provider "v1", when the current
    provider is "v2", all secrets should be successfully re-encrypted from
    v1 to v2. The secret data should remain decryptable after rotation.
    """
    # Arrange
    v1_encryptor = SecretKeyEncryptor.from_random("v1")
    v2_encryptor = SecretKeyEncryptor.from_random("v2")

    # Create secrets with actual secret data encrypted using envelope encryption
    secret_values = ["my-api-key-123", "database-password-456", "oauth-token-789"]
    secrets = [
        make_secret("secret-1", "default", v1_encryptor, secret_values[0]),
        make_secret("secret-2", "default", v1_encryptor, secret_values[1]),
        make_secret("secret-3", "workspace-a", v1_encryptor, secret_values[2]),
    ]

    entity_client = MagicMock()
    entity_client.list = AsyncMock(return_value=make_list_response(secrets))
    entity_client.update = AsyncMock()

    mocker.patch(
        "nhx.core.secrets.api.v2.admin.routines.get_encryptor_by_name",
        return_value=v1_encryptor,
    )

    # Act
    rotated_count = await rotate_encryption_keys(v2_encryptor, entity_client)

    # Assert
    assert rotated_count == 3
    assert entity_client.update.call_count == 3

    # Verify each secret was re-encrypted and data is still decryptable
    for i, secret in enumerate(secrets):
        assert secret._secret_provider == "v2"
        # The secret data should be decryptable using the new KEK (v2)
        decrypted_data = envelope_decrypt(v2_encryptor, secret._data, secret._encrypted_dek, secret._secret_provider)
        assert decrypted_data == secret_values[i]


@pytest.mark.asyncio
async def test_rotate_encryption_keys_idempotent_partial_migration(mocker):
    """Test that rotation is idempotent when some secrets are already migrated.

    If a previous rotation was interrupted, some secrets may already be on the
    v2 provider while others are still on v1. The rotation should only process
    secrets that need migration (v1 secrets) and skip those already on v2.
    Secret data should remain decryptable for all secrets after rotation.
    """
    # Arrange
    v1_encryptor = SecretKeyEncryptor.from_random("v1")
    v2_encryptor = SecretKeyEncryptor.from_random("v2")

    # Mixed state: some secrets already migrated to v2, others still on v1
    secret_values = ["already-migrated-1", "needs-migration-1", "already-migrated-2", "needs-migration-2"]
    secrets = [
        make_secret("secret-1", "default", v2_encryptor, secret_values[0]),  # Already migrated
        make_secret("secret-2", "default", v1_encryptor, secret_values[1]),  # Needs migration
        make_secret("secret-3", "workspace-a", v2_encryptor, secret_values[2]),  # Already migrated
        make_secret("secret-4", "workspace-a", v1_encryptor, secret_values[3]),  # Needs migration
    ]
    # Store original encrypted DEKs for v2 secrets to verify they're unchanged
    original_v2_encrypted_deks = [secrets[0]._encrypted_dek, secrets[2]._encrypted_dek]

    entity_client = MagicMock()
    entity_client.list = AsyncMock(return_value=make_list_response(secrets))
    entity_client.update = AsyncMock()

    mocker.patch(
        "nhx.core.secrets.api.v2.admin.routines.get_encryptor_by_name",
        return_value=v1_encryptor,
    )

    # Act
    rotated_count = await rotate_encryption_keys(v2_encryptor, entity_client)

    # Assert - only secrets on v1 should be updated
    assert rotated_count == 2
    assert entity_client.update.call_count == 2

    # Verify v2 secrets were not modified
    assert secrets[0]._secret_provider == "v2"
    assert secrets[0]._encrypted_dek == original_v2_encrypted_deks[0]
    assert secrets[2]._secret_provider == "v2"
    assert secrets[2]._encrypted_dek == original_v2_encrypted_deks[1]

    # Verify all secrets can be decrypted with v2
    for i, secret in enumerate(secrets):
        assert secret._secret_provider == "v2"
        decrypted_data = envelope_decrypt(v2_encryptor, secret._data, secret._encrypted_dek, secret._secret_provider)
        assert decrypted_data == secret_values[i]


@pytest.mark.asyncio
async def test_rotate_encryption_keys_rollback_to_v1(mocker):
    """Test that rotation can go from v2 back to v1 (reversible).

    Given access to the v2 provider's KEK, we should be able to roll back
    secrets from v2 to v1. This ensures the operation is reversible and
    secret data remains decryptable after rollback.
    """
    # Arrange
    v1_encryptor = SecretKeyEncryptor.from_random("v1")
    v2_encryptor = SecretKeyEncryptor.from_random("v2")

    # All secrets are on v2, we want to roll back to v1
    secret_values = ["rollback-secret-1", "rollback-secret-2", "rollback-secret-3"]
    secrets = [
        make_secret("secret-1", "default", v2_encryptor, secret_values[0]),
        make_secret("secret-2", "default", v2_encryptor, secret_values[1]),
        make_secret("secret-3", "workspace-a", v2_encryptor, secret_values[2]),
    ]

    entity_client = MagicMock()
    entity_client.list = AsyncMock(return_value=make_list_response(secrets))
    entity_client.update = AsyncMock()

    mocker.patch(
        "nhx.core.secrets.api.v2.admin.routines.get_encryptor_by_name",
        return_value=v2_encryptor,
    )

    # Act - v1_encryptor is now the "current" encryptor (target)
    rotated_count = await rotate_encryption_keys(v1_encryptor, entity_client)

    # Assert
    assert rotated_count == 3
    assert entity_client.update.call_count == 3

    # Verify each secret was rolled back to v1 and data is still decryptable
    for i, secret in enumerate(secrets):
        assert secret._secret_provider == "v1"
        # The secret data should be decryptable using v1 KEK
        decrypted_data = envelope_decrypt(v1_encryptor, secret._data, secret._encrypted_dek, secret._secret_provider)
        assert decrypted_data == secret_values[i]


@pytest.mark.asyncio
async def test_rotate_encryption_keys_no_secrets_to_migrate(mocker):
    """Test that rotation handles the case where all secrets are already migrated."""
    # Arrange
    v2_encryptor = SecretKeyEncryptor.from_random("v2")

    # All secrets already on v2
    secret_values = ["already-on-v2-secret-1", "already-on-v2-secret-2"]
    secrets = [
        make_secret("secret-1", "default", v2_encryptor, secret_values[0]),
        make_secret("secret-2", "default", v2_encryptor, secret_values[1]),
    ]

    entity_client = MagicMock()
    entity_client.list = AsyncMock(return_value=make_list_response(secrets))
    entity_client.update = AsyncMock()

    # Act
    rotated_count = await rotate_encryption_keys(v2_encryptor, entity_client)

    # Assert - no updates should occur
    assert rotated_count == 0
    assert entity_client.update.call_count == 0


@pytest.mark.asyncio
async def test_rotate_encryption_keys_empty_secrets_list(mocker):
    """Test that rotation handles an empty secrets list gracefully."""
    # Arrange
    v2_encryptor = SecretKeyEncryptor.from_random("v2")

    entity_client = MagicMock()
    entity_client.list = AsyncMock(return_value=make_list_response([]))
    entity_client.update = AsyncMock()

    # Act
    rotated_count = await rotate_encryption_keys(v2_encryptor, entity_client)

    # Assert
    assert rotated_count == 0
    assert entity_client.update.call_count == 0


@pytest.mark.asyncio
async def test_rotate_encryption_keys_multiple_old_providers(mocker):
    """Test that rotation handles secrets from multiple old providers.

    If secrets exist from multiple historical providers (v1, legacy, etc.),
    the rotation should correctly instantiate each old encryptor and migrate
    all secrets to the current provider. Secret data should remain decryptable.
    """
    # Arrange
    v1_encryptor = SecretKeyEncryptor.from_random("v1")
    legacy_encryptor = SecretKeyEncryptor.from_random("legacy")
    v2_encryptor = SecretKeyEncryptor.from_random("v2")

    # Secrets from multiple providers
    secret_values = ["v1-secret-data", "legacy-secret-data", "v1-secret-data-2", "legacy-secret-data-2"]
    secrets = [
        make_secret("secret-1", "default", v1_encryptor, secret_values[0]),
        make_secret("secret-2", "default", legacy_encryptor, secret_values[1]),
        make_secret("secret-3", "workspace-a", v1_encryptor, secret_values[2]),
        make_secret("secret-4", "workspace-a", legacy_encryptor, secret_values[3]),
    ]

    entity_client = MagicMock()
    entity_client.list = AsyncMock(return_value=make_list_response(secrets))
    entity_client.update = AsyncMock()

    def get_encryptor_by_name(name: str) -> SecretKeyEncryptor:
        if name == "v1":
            return v1_encryptor
        elif name == "legacy":
            return legacy_encryptor
        raise ValueError(f"Unknown provider: {name}")

    mocker.patch(
        "nhx.core.secrets.api.v2.admin.routines.get_encryptor_by_name",
        side_effect=get_encryptor_by_name,
    )

    # Act
    rotated_count = await rotate_encryption_keys(v2_encryptor, entity_client)

    # Assert
    assert rotated_count == 4
    assert entity_client.update.call_count == 4

    # All secrets should now be on v2 and data should be decryptable
    for i, secret in enumerate(secrets):
        assert secret._secret_provider == "v2"
        decrypted_data = envelope_decrypt(v2_encryptor, secret._data, secret._encrypted_dek, secret._secret_provider)
        assert decrypted_data == secret_values[i]


@pytest.mark.asyncio
async def test_rotate_encryption_keys_partial_rollback_after_partial_rollforward(mocker):
    """Test partial rollback after a partial roll forward.

    This test simulates a complex real-world scenario:
    1. Start with secrets on v1
    2. Partially migrate to v2 (interrupted - some on v1, some on v2)
    3. Decision to rollback to v1
    4. Partial rollback occurs (interrupted - mixed state)
    5. Complete the rollback to v1

    This ensures the system can handle arbitrary mixed states and recover
    in either direction.
    """
    # Arrange
    v1_encryptor = SecretKeyEncryptor.from_random("v1")
    v2_encryptor = SecretKeyEncryptor.from_random("v2")

    secret_values = ["secret-a", "secret-b", "secret-c", "secret-d"]

    # Simulate the state after partial rollforward then partial rollback:
    # - secret-1: was on v1, migrated to v2, rolled back to v1 (on v1)
    # - secret-2: was on v1, migrated to v2, not yet rolled back (on v2)
    # - secret-3: was on v1, never migrated to v2 (on v1)
    # - secret-4: was on v1, migrated to v2, rolled back to v1 (on v1)
    secrets = [
        make_secret("secret-1", "default", v1_encryptor, secret_values[0]),  # Already rolled back
        make_secret("secret-2", "default", v2_encryptor, secret_values[1]),  # Needs rollback
        make_secret("secret-3", "workspace-a", v1_encryptor, secret_values[2]),  # Never migrated
        make_secret("secret-4", "workspace-a", v1_encryptor, secret_values[3]),  # Already rolled back
    ]

    entity_client = MagicMock()
    entity_client.list = AsyncMock(return_value=make_list_response(secrets))
    entity_client.update = AsyncMock()

    mocker.patch(
        "nhx.core.secrets.api.v2.admin.routines.get_encryptor_by_name",
        return_value=v2_encryptor,
    )

    # Act - complete the rollback to v1
    rotated_count = await rotate_encryption_keys(v1_encryptor, entity_client)

    # Assert - only the secret still on v2 should be updated
    assert rotated_count == 1
    assert entity_client.update.call_count == 1

    # Verify all secrets are now on v1 and data is decryptable
    for i, secret in enumerate(secrets):
        assert secret._secret_provider == "v1"
        decrypted_data = envelope_decrypt(v1_encryptor, secret._data, secret._encrypted_dek, secret._secret_provider)
        assert decrypted_data == secret_values[i]


@pytest.mark.asyncio
async def test_rotate_encryption_keys_with_pagination_large_dataset(mocker):
    """Test rotation with a large dataset that exceeds the default page size.

    This test creates 999 secrets (333 per workspace across 3 workspaces) to
    verify that pagination works correctly when there are more secrets than
    fit in a single page (default page_size=100). This results in 10 pages
    of data that must all be processed.
    """
    # Arrange
    v1_encryptor = SecretKeyEncryptor.from_random("v1")
    v2_encryptor = SecretKeyEncryptor.from_random("v2")

    workspaces = ["workspace-alpha", "workspace-beta", "workspace-gamma"]
    secrets_per_workspace = 333
    total_secrets = len(workspaces) * secrets_per_workspace  # 999 secrets

    # Create secrets distributed across workspaces
    secrets = []
    secret_values = []
    for workspace in workspaces:
        for i in range(secrets_per_workspace):
            secret_value = f"secret-data-{workspace}-{i:03d}"
            secret_values.append(secret_value)
            secrets.append(make_secret(f"secret-{i:03d}", workspace, v1_encryptor, secret_value))

    # Create paginated responses (page_size=100 matches the routine's page_size)
    paginated_responses = make_paginated_list_responses(secrets, page_size=100)

    def mock_list(*args, **kwargs):
        """Return the appropriate page based on the page parameter."""
        page = kwargs.get("page", 1)
        # Adjust for 1-indexed pages
        idx = page - 1
        if idx < len(paginated_responses):
            return paginated_responses[idx]
        # Return empty last page if we go past
        return ListResponse(
            data=[],
            pagination=PaginationInfo(
                page=page,
                page_size=100,
                current_page_size=0,
                total_pages=len(paginated_responses),
                total_results=total_secrets,
            ),
        )

    entity_client = MagicMock()
    entity_client.list = AsyncMock(side_effect=mock_list)
    entity_client.update = AsyncMock()

    mocker.patch(
        "nhx.core.secrets.api.v2.admin.routines.get_encryptor_by_name",
        return_value=v1_encryptor,
    )

    # Act
    rotated_count = await rotate_encryption_keys(v2_encryptor, entity_client)

    # Assert - all 999 secrets should be updated
    assert rotated_count == total_secrets
    assert entity_client.update.call_count == total_secrets

    # Verify pagination was used correctly (should have called list 10 times for 999 secrets)
    expected_pages = (total_secrets + 99) // 100  # ceil(999/100) = 10
    assert entity_client.list.call_count == expected_pages

    # Verify all secrets were re-encrypted and data is still decryptable
    for i, secret in enumerate(secrets):
        assert secret._secret_provider == "v2", f"Secret {i} not migrated to v2"
        decrypted_data = envelope_decrypt(v2_encryptor, secret._data, secret._encrypted_dek, secret._secret_provider)
        assert decrypted_data == secret_values[i], f"Secret {i} data mismatch after rotation"

    # Verify secrets are distributed across workspaces as expected
    workspace_counts = {ws: 0 for ws in workspaces}
    for secret in secrets:
        workspace_counts[secret.workspace] += 1
    for workspace in workspaces:
        assert workspace_counts[workspace] == secrets_per_workspace

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from nhx.common.secrets.encryption import (
    SecretKeyEncryptor,
    SecretKeyEncryptorConfig,
    envelope_decrypt,
    envelope_encrypt,
)
from nhx.core.secrets.entities import HelixSecret
from pydantic import SecretStr


async def test_list_secrets_entities_empty(client_context):
    workspace = "default"
    response = await client_context.entity_client.list(HelixSecret, workspace=workspace, page_size=10)
    assert response.data == []
    assert response.pagination.total_results == 0


async def test_access_old_secret_with_old_provider_can_be_accessed(
    old_encryption_key, current_encryption_key, client_context
):
    """Test creating a secret with an old provider and retrieving it. This case verifies that we can work with existing secrets while migrating to a new encryptor."""

    secret_name = "entity-client-secret"
    secret_value = "supersecret"

    # Get an encryptor that matches the old provider
    old_encryptor = SecretKeyEncryptor.from_config(
        name="old", config=SecretKeyEncryptorConfig(value=old_encryption_key)
    )

    # Envelope encrypt the secret value
    encrypted_data, encrypted_dek, provider_name = envelope_encrypt(old_encryptor, secret_value)
    secret = HelixSecret(
        name=secret_name,
        workspace="default",
        description="Test secret created via EntityClient",
    )
    secret._data = encrypted_data
    secret._encrypted_dek = encrypted_dek
    secret._secret_provider = provider_name

    created_secret = await client_context.entity_client.create(secret)
    assert created_secret.name == secret_name
    assert created_secret.id is not None

    secrets = client_from_platform(client_context.sdk, SecretsClient)

    # Retrieve the secret through the API and validate that it can be decrypted correctly
    retrieved_secret = secrets.get_secret(name=secret_name, workspace="default").data()
    assert retrieved_secret.name == secret_name

    # Access the secret value and verify decryption
    accessed_secret = secrets.access_secret(name=secret_name, workspace="default").data()
    assert accessed_secret.name == secret_name
    assert accessed_secret.value == secret_value

    # Now create a new secret with the current provider to ensure both can coexist
    new_secret_name = "entity-client-new-secret"
    new_secret_value = "newsupersecret"
    new_created_secret = secrets.create_secret(
        body=HelixSecretCreateRequest(
            name=new_secret_name,
            value=SecretStr(new_secret_value),
            description="New secret with current provider",
        ),
        workspace="default",
    ).data()
    assert new_created_secret.name == new_secret_name

    # Access the new secret and verify decryption
    new_accessed_secret = secrets.access_secret(name=new_secret_name, workspace="default").data()
    assert new_accessed_secret.name == new_secret_name
    assert new_accessed_secret.value == new_secret_value

    # Now, access the new secret via the entity client, and try decrypting with the older provider. This should fail.
    retrieved_new_secret = await client_context.entity_client.get(
        HelixSecret, workspace="default", name=new_secret_name
    )
    assert retrieved_new_secret.name == new_secret_name

    with pytest.raises(Exception):
        # Attempt to decrypt with the old encryptor, which should fail
        envelope_decrypt(
            old_encryptor,
            retrieved_new_secret._data,
            retrieved_new_secret._encrypted_dek,
            retrieved_new_secret._secret_provider,
        )

    # But, it should work with the current encryptor
    current_encryptor = SecretKeyEncryptor.from_config(
        name="current", config=SecretKeyEncryptorConfig(value=current_encryption_key)
    )
    decrypted_data = envelope_decrypt(
        current_encryptor,
        retrieved_new_secret._data,
        retrieved_new_secret._encrypted_dek,
        retrieved_new_secret._secret_provider,
    )
    assert decrypted_data == new_secret_value

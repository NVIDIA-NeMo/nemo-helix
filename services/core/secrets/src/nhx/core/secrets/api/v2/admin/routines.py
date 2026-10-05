# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from nhx.common.entities import ALL_WORKSPACES, EntityClient
from nhx.common.secrets.encryption import HelixEncryptor
from nhx.core.secrets.app.encryptor import get_encryptor_by_name
from nhx.core.secrets.entities import HelixSecret

logger = logging.getLogger(__name__)


async def rotate_encryption_keys(current_encryptor: HelixEncryptor, entity_client: EntityClient) -> int:
    """
    Rotate encryption keys for the platform encryptor.

    This routine iterates over all platform secrets, decrypts their data using
    the old encryption key, and re-encrypts it using the current encryption key.
    It updates each secret's encrypted data key (DEK) and secret provider accordingly.

    Args:
        current_encryptor (HelixEncryptor): The current encryptor to use for re-encryption.
        entity_client (EntityClient): The entity client to interact with platform secrets.
    Returns:
        int: The number of secrets that were rotated.
    """
    logger.warning("Initiating rotation of encryption keys")
    current_provider = current_encryptor.name
    providers: dict[str, HelixEncryptor] = dict()

    page = 1
    page_size = 100
    total_pages = 1  # Will be updated after first fetch

    total_secrets_rotated = 0

    while page <= total_pages:
        secrets = await entity_client.list(
            entity_type=HelixSecret, workspace=ALL_WORKSPACES, page=page, page_size=page_size
        )
        total_pages = secrets.pagination.total_pages

        for secret in secrets.data:
            if secret._secret_provider == current_provider:
                logger.debug("Secret '%s/%s' already using current provider, skipping", secret.workspace, secret.name)
                continue  # already using current provider

            if secret._secret_provider not in providers:
                logger.debug(
                    f"Instantiating encryptor for provider '{secret._secret_provider}' for secret re-encryption"
                )
                providers[secret._secret_provider] = get_encryptor_by_name(secret._secret_provider)

            logger.debug(
                "Re-encrypting secret '%s/%s' from provider '%s' to '%s'",
                secret.workspace,
                secret.name,
                secret._secret_provider,
                current_provider,
            )
            old_encryptor = providers[secret._secret_provider]
            # Decrypt DEK with old encryptor
            encrypted_dek = secret._encrypted_dek
            dek = old_encryptor.decrypt(encrypted_dek)
            # Encrypt DEK with new encryptor
            reencrypted_dek = current_encryptor.encrypt(dek)

            # Update the secret and save with new encrypted DEK and provider
            secret._encrypted_dek = reencrypted_dek
            secret._secret_provider = current_provider
            await entity_client.update(secret)
            total_secrets_rotated += 1
            logger.debug("Successfully rotated encryption key for secret '%s/%s'", secret.workspace, secret.name)

        page += 1

    logger.info("Completed rotation of encryption keys for all secrets")
    return total_secrets_rotated

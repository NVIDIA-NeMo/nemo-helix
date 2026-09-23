# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path

from nhx.common.config import get_service_config
from nhx.common.secrets.encryption import (
    HelixEncryptor,
    SecretKeyEncryptor,
    SecretKeyEncryptorConfig,
    VaultEncryptor,
    VaultEncryptorConfig,
    get_base64_encoded_random_bytes,
)
from nhx.core.secrets.config import SecretsServiceConfig

logger = logging.getLogger(__name__)


def local_key_creation() -> HelixEncryptor:
    """Create a local secret key encryption provider."""
    config = get_service_config(SecretsServiceConfig)
    if config.allow_key_creation:
        path = config.local_key_creation_path
        if not os.path.exists(path):
            # Persist a new random key to the file
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            random_key = get_base64_encoded_random_bytes(32)
            with open(path, "w") as f:
                f.write(random_key)
            logger.info(f"Created new random key and persisted to {path} for local secret key encryption provider.")

        return SecretKeyEncryptor.from_file(name="", path=path)
    raise ValueError("Cannot create local secret key encryption provider. allow_key_creation is disabled.")


def get_encryptor_by_name(name: str) -> HelixEncryptor:
    """Get an encryptor by name."""
    config = get_service_config(SecretsServiceConfig)
    # Local key creation: empty name and allow_key_creation use file-based key
    if config.allow_key_creation and name == "":
        return local_key_creation()
    encryptor_config = config.encryption.providers.get_provider_config(name)
    if isinstance(encryptor_config, SecretKeyEncryptorConfig):
        return SecretKeyEncryptor.from_config(config=encryptor_config, name=name)
    elif isinstance(encryptor_config, VaultEncryptorConfig):
        return VaultEncryptor.from_config(config=encryptor_config, name=name)
    else:
        raise ValueError(f"Cannot get encryptor configuration for provider {name}.")


def get_current_encryptor() -> HelixEncryptor:
    """Get the current encryptor based on service configuration."""
    config = get_service_config(SecretsServiceConfig)
    return get_encryptor_by_name(config.encryption.current_provider)

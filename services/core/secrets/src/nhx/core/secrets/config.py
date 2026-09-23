# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.common.config import create_service_config_class, internal_field, nhx_user_data_dir
from nhx.common.secrets.encryption import SecretKeyEncryptorConfig, VaultEncryptorConfig
from pydantic import BaseModel, Field, model_validator


class ProviderConfig(BaseModel):
    secret_key: dict[str, SecretKeyEncryptorConfig] = Field(
        default_factory=dict,
        description="Mapping of secret key encryptor names to their configurations.",
    )
    vault: dict[str, VaultEncryptorConfig] = Field(
        default_factory=dict,
        description="Mapping of vault encryptor names to their configurations.",
    )

    @model_validator(mode="after")
    def validate(self):
        """Validate that all provider names are unique across types."""
        all_names = set()
        for name in self.secret_key.keys():
            if name in all_names:
                raise ValueError(f"Duplicate encryptor name found: {name}")
            all_names.add(name)
        for name in self.vault.keys():
            if name in all_names:
                raise ValueError(f"Duplicate encryptor name found: {name}")
            all_names.add(name)
        return self

    def get_provider_config(self, name: str) -> SecretKeyEncryptorConfig | VaultEncryptorConfig | None:
        """Retrieve the provider configuration by name.

        Args:
            name: The name of the provider (e.g., "v1" or "v2").

        Returns:
            The corresponding SecretKeyEncryptorConfig or VaultEncryptorConfig.

        Raises:
            ValueError: If no provider with the given name exists.
        """
        if name in self.secret_key:
            return self.secret_key[name]
        if name in self.vault:
            return self.vault[name]
        raise ValueError(f"No encryptor configuration found with name: {name}")


class EncryptionConfig(BaseModel):
    current_provider: str = Field(
        default="",
        description="The type of encryption key used to encrypt/decrypt secrets.",
    )
    providers: ProviderConfig = Field(
        default_factory=ProviderConfig,
        description="Configuration for all available encryption providers.",
    )


class SecretsServiceConfig(create_service_config_class("secrets")):  # type: ignore
    """
    Configuration for the Secrets service.
    """

    encryption: EncryptionConfig = Field(
        default_factory=EncryptionConfig,
        description="Encryption configuration for the Secrets service.",
    )
    allow_key_creation: bool = internal_field(
        default=False,
        description=(
            "Whether to allow random secret key encryption creation. This is insecure and should only be used for development/demonstration purposes. "
            + "Setting this to true will allow the creation of a local secret key encryption provider, and is only enabled if encryption.current_provider is not set."
        ),
    )
    local_key_creation_path: str = internal_field(
        default_factory=lambda: str(nhx_user_data_dir() / "nhx-encryption-key.txt"),
        description="Path to the file where the random secret key encryption key will be persisted. This is only used if allow_key_creation is true.",
    )

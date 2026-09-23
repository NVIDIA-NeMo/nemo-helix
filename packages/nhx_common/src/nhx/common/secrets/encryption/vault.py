# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64

import hvac
from hvac.exceptions import VaultError
from nhx.common.secrets.encryption.base import HelixEncryptor
from nhx.common.secrets.exceptions import EncryptionError
from pydantic import BaseModel, Field


class VaultEncryptorConfig(BaseModel):
    """Configuration for the VaultEncryptor."""

    key_name: str = Field(
        default="nemo-helix-key", description="Name of the key in Vault to use for encryption/decryption."
    )
    address: str | None = Field(
        default=None,
        description="Address of the Vault server. If not specificed, the value of the VAULT_ADDR env variable will be used.",
    )
    token: str | None = Field(
        default=None,
        description="Authentication token for Vault. If not specified, the value of the VAULT_TOKEN env variable or the content of the file located at ~/.vault-token will be used.",
    )


class VaultEncryptor(HelixEncryptor):
    def __init__(self, name: str, key_name: str, *, address: str | None, token: str | None):
        super().__init__(name=name)
        self._key_name = key_name
        self._client = hvac.Client(url=address, token=token)
        if not self._client.is_authenticated():
            raise ValueError("Vault client failed to authenticate. Check your Vault address and/or token.")

    @classmethod
    def from_config(cls, name: str, config: VaultEncryptorConfig) -> VaultEncryptor:
        return cls(name=name, key_name=config.key_name, address=config.address, token=config.token)

    def encrypt(self, plaintext: str) -> str:
        plaintext_b64 = base64.b64encode(plaintext.encode()).decode()
        try:
            response = self._client.secrets.transit.encrypt_data(name=self._key_name, plaintext=plaintext_b64)
            return response["data"]["ciphertext"]
        except VaultError as e:
            raise EncryptionError(f"Encryption failed: {str(e)}") from e

    def decrypt(self, encrypted_data: str) -> str:
        try:
            response = self._client.secrets.transit.decrypt_data(name=self._key_name, ciphertext=encrypted_data)
            plaintext_b64 = response["data"]["plaintext"]
            plaintext_bytes = base64.b64decode(plaintext_b64)
            return plaintext_bytes.decode()
        except VaultError as e:
            raise EncryptionError(f"Decryption failed: {str(e)}") from e

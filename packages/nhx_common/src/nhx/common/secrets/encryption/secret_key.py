# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Optional

from nhx.common.secrets.encryption.aes256 import decrypt, encrypt
from nhx.common.secrets.encryption.base import (
    PlatformEncryptor,
    get_base64_encoded_random_bytes,
)
from nhx.common.secrets.exceptions import EncryptionError
from pydantic import BaseModel, Field, PrivateAttr, model_validator

logger = logging.getLogger(__name__)


class SecretKeyEncryptorConfig(BaseModel):
    """Configuration for the SecretKeyEncryptor."""

    value: Optional[str] = Field(
        default=None,
        description="Base64-encoded encryption key",
    )
    from_env: Optional[str] = Field(
        default=None,
        description="Indicates if the key should be loaded from an environment variable. If set, 'value' is ignored.",
    )
    _key: bytes = PrivateAttr()

    @model_validator(mode="after")
    def validate_key(self):
        """Validate and decode the base64-encoded key.

        Returns:
            The decoded key as bytes.

        Raises:
            ValueError: If the key is not valid base64 or does not meet length requirements.
        """
        if self.value is None and self.from_env is None:
            raise ValueError("Encryption key value must be provided from 'value' or 'from_env'")
        try:
            if self.from_env is not None:
                env_value = os.getenv(self.from_env)
                if env_value is None:
                    raise ValueError(
                        f"Environment variable for secret_key encryption provider '{self.from_env}' is not set"
                    )
                self._key = base64.b64decode(env_value)
            else:
                if self.value is None:
                    raise ValueError("Encryption key value must be provided")
                self._key = base64.b64decode(self.value)
            if len(self._key) < 32:
                raise ValueError("Encryption key must be at least 32 bytes long")
            return self
        except Exception as e:
            raise ValueError("Invalid base64-encoded encryption key") from e


class SecretKeyEncryptor(PlatformEncryptor):
    """
    Platform Encryptor that uses a secret key for AES-256 encryption.
    This encryptor derives an AES-256 key from the provided secret key using HKDF.
    """

    def __init__(self, name: str, key: bytes):
        """Initialize the SecretKeyEncryptor with a key

        Args:
            key: A password-like secret key.
        """
        super().__init__(name=name)
        if not key or len(key) < 32:
            raise ValueError("Encryption key must be at least 32 bytes long")
        self.key = key

    @classmethod
    def from_config(cls, name: str, config: SecretKeyEncryptorConfig) -> SecretKeyEncryptor:
        """Create an SecretKeyEncryptor instance from configuration.

        Args:
            config: SecretKeyEncryptorConfig instance with local key
        Returns:
            An initialized SecretKeyEncryptor instance.
        """
        return cls(name=name, key=config._key)

    @classmethod
    def from_random(cls, name: str) -> SecretKeyEncryptor:
        """Create a SecretKeyEncryptor instance with a random key.

        Returns:
            An initialized SecretKeyEncryptor instance with a random key.
        """
        random_key = get_base64_encoded_random_bytes(32)
        config = SecretKeyEncryptorConfig(value=random_key)
        return cls.from_config(name=name, config=config)

    @classmethod
    def from_file(cls, name: str, path: str | Path) -> SecretKeyEncryptor:
        """Create a SecretKeyEncryptor instance from a file containing a base64-encoded key.

        Args:
            name: Provider name (often "" for local key creation).
            path: Path to a file containing the base64-encoded key (one line).

        Returns:
            An initialized SecretKeyEncryptor instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Encryption key file not found: {path}")
        content = path.read_text().strip()
        config = SecretKeyEncryptorConfig(value=content)
        return cls.from_config(name=name, config=config)

    def encrypt(self, plaintext: str) -> str:
        try:
            encrypted_bytes = encrypt(plaintext.encode(), self.key)
            return base64.b64encode(encrypted_bytes).decode()
        except Exception as e:
            logger.error("Encryption failed: %s", e)
            raise EncryptionError("Encryption failed") from e

    def decrypt(self, encrypted_data: str) -> str:
        try:
            encrypted_bytes = base64.b64decode(encrypted_data)
            return decrypt(encrypted_bytes, self.key).decode()
        except Exception as e:
            logger.error("Decryption failed: %s", e)
            raise EncryptionError("Decryption failed") from e

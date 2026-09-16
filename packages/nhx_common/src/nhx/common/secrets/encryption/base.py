# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
import os

logger = logging.getLogger(__name__)


class PlatformEncryptor:
    """Base class for encryption handlers."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        """Return the name of the encryptor."""
        return self._name

    def encrypt(self, plaintext: str) -> str:
        """Encrypt the given plaintext string.

        Args:
            plaintext: The string to encrypt.

        Returns:
            The encrypted string.
        """
        raise NotImplementedError("Encrypt method not implemented")

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt the given encrypted string.

        Args:
            encrypted_data: The string to decrypt.

        Returns:
            The decrypted plaintext string.
        """
        raise NotImplementedError("Decrypt method not implemented")


def get_random_bytes(length: int) -> bytes:
    return os.urandom(length)


def get_base64_encoded_random_bytes(length: int) -> str:
    return base64.b64encode(get_random_bytes(length)).decode()

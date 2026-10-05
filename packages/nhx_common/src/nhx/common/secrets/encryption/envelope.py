# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from typing import Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from nhx.common.secrets.encryption.base import HelixEncryptor, get_random_bytes

_AES_KEY_SIZE = 32  # 256 bits
_NONCE_LENGTH = 12


def envelope_encrypt(kek_encryptor: HelixEncryptor, plaintext: str) -> Tuple[str, str, str]:
    """Perform envelope encryption on the given plaintext.

    Args:
        kek_encryptor: The key encryption key (KEK) encryptor.
        plaintext: The plaintext data to encrypt.

    Returns:
        A tuple containing the encrypted data and the encrypted data encryption key (DEK).
    """
    # Generate a new data encryption key (DEK) for this secret
    dek = get_random_bytes(_AES_KEY_SIZE)
    # Use the DEK to encrypt the secret data
    data_encryptor = AESGCM(dek)
    nonce = get_random_bytes(_NONCE_LENGTH)
    encrypted_bytes = data_encryptor.encrypt(nonce, plaintext.encode(), None)
    encrypted_data = base64.b64encode(nonce + encrypted_bytes).decode()
    # Encrypt the DEK with the platform's key encryption key (KEK)
    encrypted_dek = kek_encryptor.encrypt(base64.b64encode(dek).decode())
    return encrypted_data, encrypted_dek, kek_encryptor.name


def envelope_decrypt(
    kek_encryptor: HelixEncryptor, encrypted_data: str, encrypted_dek: str, secret_provider: str
) -> str:
    """Perform envelope decryption on the given encrypted data.

    Args:
        kek_encryptor: The key encryption key (KEK) encryptor.
        encrypted_data: The encrypted data to decrypt.
        encrypted_dek: The encrypted data encryption key (DEK).
        secret_provider: The name of the secret provider used for encryption.
    Returns:
        The decrypted plaintext data.
    Raises:
        ValueError: If the secret provider does not match the KEK encryptor name.
    """
    if kek_encryptor.name != secret_provider:
        raise ValueError("Mismatched secret provider for decryption")
    # Decrypt the DEK with the platform's key encryption key (KEK)
    dek = base64.b64decode(kek_encryptor.decrypt(encrypted_dek))
    # Use the DEK to decrypt the secret data
    data_encryptor = AESGCM(dek)
    encrypted_bytes = base64.b64decode(encrypted_data)
    nonce = encrypted_bytes[:_NONCE_LENGTH]
    ciphertext = encrypted_bytes[_NONCE_LENGTH:]
    plaintext = data_encryptor.decrypt(nonce, ciphertext, None)
    return plaintext.decode()

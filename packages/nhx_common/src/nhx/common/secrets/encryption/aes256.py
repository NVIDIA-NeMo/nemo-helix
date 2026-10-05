# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from nhx.common.secrets.encryption.base import get_random_bytes

"""
The construction of an AES-256 encrypted payload used in NeMo Helix is as follows:

        Salt       Nonce          Encrypted
        |            |             Payload
        |            |                |
        |  +---------v-------------+  |
        +-->SSSSSSSNNNNNNNEEEEEEEEE<--+
           +-----------------------+

Where:
- SSSSSSS is a 16-byte salt used for key derivation.
- NNNNNNN is a 12-byte nonce used for AES-GCM encryption.
- EEEEEEEEE is the AES-256-GCM encrypted payload.

This allows for secure encryption and decryption of data using a password-derived key.
"""

_SALT_LENGTH = 16
_NONCE_LENGTH = 12


def derive_aes256_cipher_key(key: bytes, salt: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=None,
    ).derive(key)


def encrypt(plaintext: bytes, key: bytes) -> bytes:
    salt = get_random_bytes(_SALT_LENGTH)
    nonce = get_random_bytes(_NONCE_LENGTH)
    cipher_key = derive_aes256_cipher_key(key, salt)
    aesgcm = AESGCM(cipher_key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return salt + nonce + ciphertext


def decrypt(encrypted_payload: bytes, key: bytes) -> bytes:
    salt = encrypted_payload[:_SALT_LENGTH]
    nonce = encrypted_payload[_SALT_LENGTH : _SALT_LENGTH + _NONCE_LENGTH]
    ciphertext = encrypted_payload[_SALT_LENGTH + _NONCE_LENGTH :]
    cipher_key = derive_aes256_cipher_key(key, salt)
    aesgcm = AESGCM(cipher_key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext

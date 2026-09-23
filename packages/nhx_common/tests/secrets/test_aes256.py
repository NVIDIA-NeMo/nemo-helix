# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.common.secrets.encryption.aes256 import decrypt, encrypt
from nhx.common.secrets.encryption.base import get_random_bytes


def test_aes256_encryption_decryption_roundtrip():
    """Test that AES-256 encryption and decryption produce the original value."""
    plaintext = b"This is a secret message"
    key = get_random_bytes(32)  # 32 bytes for AES-256
    ciphertext = encrypt(plaintext, key)
    decrypted = decrypt(ciphertext, key)
    assert decrypted == plaintext


def test_envelope_aes256_encryption_decryption():
    """Test that AES-256 envelope encryption and decryption work correctly."""
    data = b"Another secret message"

    data_encryption_key = get_random_bytes(32)
    encrypted_data = encrypt(data, data_encryption_key)

    kek_key = get_random_bytes(32)
    encrypted_data_encryption_key = encrypt(data_encryption_key, kek_key)

    decrypted_data_encryption_key = decrypt(encrypted_data_encryption_key, kek_key)
    assert decrypted_data_encryption_key == data_encryption_key

    decrypted_data = decrypt(encrypted_data, decrypted_data_encryption_key)
    assert decrypted_data == data

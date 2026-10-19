# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AES encryption utilities."""

import base64

import pytest
from nhx.common.secrets.encryption import (
    SecretKeyEncryptor,
    SecretKeyEncryptorConfig,
    envelope_decrypt,
    envelope_encrypt,
)
from nhx.common.secrets.encryption.base import get_random_bytes
from nhx.common.secrets.exceptions import EncryptionError

ENCRYPTOR_NAME = "v1"
TEST_KEY = get_random_bytes(32)  # 32 bytes for AES-256
SHORT_TEST_KEY = get_random_bytes(16)  # 16 bytes, too short for encryption entropy


def test_encryption_decryption_roundtrip():
    """Test that encryption and decryption produce the original value."""
    encryptor = SecretKeyEncryptor(ENCRYPTOR_NAME, TEST_KEY)
    plaintext = "This is a secret message"
    encrypted = encryptor.encrypt(plaintext)
    decrypted = encryptor.decrypt(encrypted)
    assert decrypted == plaintext
    assert encrypted != plaintext  # Ensure it's actually encrypted


def test_encryption_produces_different_ciphertexts():
    """Test that encrypting the same plaintext twice produces different ciphertexts."""
    encryptor = SecretKeyEncryptor(ENCRYPTOR_NAME, TEST_KEY)
    plaintext = "secret"
    encrypted1 = encryptor.encrypt(plaintext)
    encrypted2 = encryptor.encrypt(plaintext)
    # Different nonces should produce different ciphertexts
    assert encrypted1 != encrypted2
    # But both should decrypt to the same plaintext
    assert encryptor.decrypt(encrypted1) == plaintext
    assert encryptor.decrypt(encrypted2) == plaintext


def test_decrypt_with_wrong_key_raises_error():
    """Test that decrypting with wrong key raises an error."""
    encryptor1 = SecretKeyEncryptor(ENCRYPTOR_NAME, TEST_KEY)
    encryptor2 = SecretKeyEncryptor(ENCRYPTOR_NAME, get_random_bytes(32))  # Different key
    encrypted = encryptor1.encrypt("secret")
    with pytest.raises(EncryptionError):
        encryptor2.decrypt(encrypted)


def test_decrypt_corrupted_data_raises_error():
    """Test that decrypting corrupted data raises an error."""
    encryptor = SecretKeyEncryptor(ENCRYPTOR_NAME, TEST_KEY)
    with pytest.raises(EncryptionError):
        encryptor.decrypt("corrupted-data")


def test_empty_string_encryption():
    """Test encrypting and decrypting empty strings."""
    encryptor = SecretKeyEncryptor(ENCRYPTOR_NAME, TEST_KEY)
    encrypted = encryptor.encrypt("")
    decrypted = encryptor.decrypt(encrypted)
    assert decrypted == ""


def test_unicode_encryption():
    """Test encrypting and decrypting Unicode strings."""
    encryptor = SecretKeyEncryptor(ENCRYPTOR_NAME, TEST_KEY)
    plaintext = "Hello 世界 🌍 Привет"
    encrypted = encryptor.encrypt(plaintext)
    decrypted = encryptor.decrypt(encrypted)
    assert decrypted == plaintext


def test_long_text_encryption():
    """Test encrypting and decrypting long text."""
    encryptor = SecretKeyEncryptor(ENCRYPTOR_NAME, TEST_KEY)
    plaintext = "A" * 10000  # 10KB of data
    encrypted = encryptor.encrypt(plaintext)
    decrypted = encryptor.decrypt(encrypted)
    assert decrypted == plaintext


def test_envelope_encryption_decryption():
    """Test that envelope encryption and decryption work correctly."""
    kek_encryptor = SecretKeyEncryptor(ENCRYPTOR_NAME, TEST_KEY)
    plaintext = "This is a secret message for envelope encryption"
    encrypted_data, encrypted_dek, encryptor_name = envelope_encrypt(kek_encryptor, plaintext)
    decrypted_plaintext = envelope_decrypt(kek_encryptor, encrypted_data, encrypted_dek, encryptor_name)
    assert decrypted_plaintext == plaintext


def test_envelope_decryption_with_wrong_kek_raises_error():
    """Test that envelope decryption with wrong KEK raises an error."""
    kek_encryptor = SecretKeyEncryptor(ENCRYPTOR_NAME, TEST_KEY)
    wrong_kek_encryptor = SecretKeyEncryptor(ENCRYPTOR_NAME, get_random_bytes(32))  # Different key
    plaintext = "This is a secret message for envelope encryption"
    encrypted_data, encrypted_dek, encryptor_name = envelope_encrypt(kek_encryptor, plaintext)
    with pytest.raises(EncryptionError):
        envelope_decrypt(wrong_kek_encryptor, encrypted_data, encrypted_dek, encryptor_name)


def test_envelope_decryption_with_wrong_provider_raises_error():
    """Test that envelope decryption with wrong secret provider raises an error."""
    kek_encryptor = SecretKeyEncryptor(ENCRYPTOR_NAME, TEST_KEY)
    plaintext = "This is a secret message for envelope encryption"
    encrypted_data, encrypted_dek, _ = envelope_encrypt(kek_encryptor, plaintext)
    wrong_provider_name = "wrong_provider"
    with pytest.raises(ValueError):
        envelope_decrypt(kek_encryptor, encrypted_data, encrypted_dek, wrong_provider_name)


def test_secret_key_encryptor_config_validation():
    """Test that SecretKeyEncryptorConfig validates key length."""
    # Valid key
    valid_key = base64.b64encode(get_random_bytes(32)).decode()
    config = SecretKeyEncryptorConfig(value=valid_key)
    assert config._key is not None

    # Invalid key (too short)
    short_key = base64.b64encode(SHORT_TEST_KEY).decode()
    with pytest.raises(ValueError):
        SecretKeyEncryptorConfig(value=short_key)

    # Invalid base64
    invalid_base64_key = "not-a-valid-base64-string"
    with pytest.raises(ValueError):
        SecretKeyEncryptorConfig(value=invalid_base64_key)


def test_secret_key_encryptor_config_from_env(monkeypatch):
    """Test that SecretKeyEncryptorConfig can load key from environment variable."""
    # Generate a valid key and set it in the environment
    valid_key = base64.b64encode(get_random_bytes(32)).decode()
    env_var_name = "TEST_ENCRYPTION_KEY"
    monkeypatch.setenv(env_var_name, valid_key)

    # Create config using from_env
    config = SecretKeyEncryptorConfig(from_env=env_var_name)
    assert config._key is not None
    assert len(config._key) == 32

    # Verify the encryptor works with this config
    encryptor = SecretKeyEncryptor.from_config("test", config)
    plaintext = "secret message"
    encrypted = encryptor.encrypt(plaintext)
    decrypted = encryptor.decrypt(encrypted)
    assert decrypted == plaintext


def test_secret_key_encryptor_config_from_env_missing_var():
    """Test that SecretKeyEncryptorConfig raises error when env var is not set."""
    env_var_name = "NONEXISTENT_ENCRYPTION_KEY_VAR"
    with pytest.raises(ValueError):
        SecretKeyEncryptorConfig(from_env=env_var_name)


def test_secret_key_encryptor_config_from_env_invalid_key(monkeypatch):
    """Test that SecretKeyEncryptorConfig raises error when env var contains invalid key."""
    env_var_name = "TEST_INVALID_ENCRYPTION_KEY"

    # Test with invalid base64
    monkeypatch.setenv(env_var_name, "not-valid-base64!")
    with pytest.raises(ValueError):
        SecretKeyEncryptorConfig(from_env=env_var_name)


def test_secret_key_encryptor_config_from_env_short_key(monkeypatch):
    """Test that SecretKeyEncryptorConfig raises error when env var contains short key."""
    env_var_name = "TEST_SHORT_ENCRYPTION_KEY"

    # Test with key that's too short
    short_key = base64.b64encode(get_random_bytes(16)).decode()
    monkeypatch.setenv(env_var_name, short_key)
    with pytest.raises(ValueError):
        SecretKeyEncryptorConfig(from_env=env_var_name)


def test_secret_key_encryptor_config_from_env_takes_precedence(monkeypatch):
    """Test that from_env takes precedence over value when both are provided."""
    env_var_name = "TEST_PRECEDENCE_ENCRYPTION_KEY"
    env_key = get_random_bytes(32)
    value_key = get_random_bytes(32)

    # Ensure the keys are different
    assert env_key != value_key

    monkeypatch.setenv(env_var_name, base64.b64encode(env_key).decode())

    # When both from_env and value are provided, from_env should take precedence
    config = SecretKeyEncryptorConfig(
        from_env=env_var_name,
        value=base64.b64encode(value_key).decode(),
    )

    # The config should use the key from the environment variable
    assert config._key == env_key


def test_secret_key_encryptor_from_file(tmp_path):
    """Test SecretKeyEncryptor.from_file loads key from file and encrypts/decrypts."""
    key_file = tmp_path / "key.txt"
    key_b64 = base64.b64encode(get_random_bytes(32)).decode()
    key_file.write_text(key_b64)

    encryptor = SecretKeyEncryptor.from_file("local", key_file)
    plaintext = "secret from file"
    encrypted = encryptor.encrypt(plaintext)
    decrypted = encryptor.decrypt(encrypted)
    assert decrypted == plaintext


def test_secret_key_encryptor_from_file_nonexistent():
    """Test SecretKeyEncryptor.from_file raises FileNotFoundError when file does not exist."""
    with pytest.raises(FileNotFoundError, match="Encryption key file not found"):
        SecretKeyEncryptor.from_file("local", "/nonexistent/key.txt")


def test_secret_key_encryptor_from_file_strips_whitespace(tmp_path):
    """Test SecretKeyEncryptor.from_file strips surrounding whitespace from key file."""
    key_file = tmp_path / "key.txt"
    key_b64 = base64.b64encode(get_random_bytes(32)).decode()
    key_file.write_text(f"  \n{key_b64}\n  ")

    encryptor = SecretKeyEncryptor.from_file("local", key_file)
    plaintext = "data"
    assert encryptor.decrypt(encryptor.encrypt(plaintext)) == plaintext

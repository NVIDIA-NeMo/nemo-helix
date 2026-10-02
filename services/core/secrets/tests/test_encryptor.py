# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for secrets encryptor module: local key creation config and usage."""

import pytest
from nhx.common.config import Configuration, nhx_user_data_dir
from nhx.common.secrets.encryption import SecretKeyEncryptor, get_base64_encoded_random_bytes
from nhx.core.secrets.app.encryptor import get_encryptor_by_name, local_key_creation
from nhx.core.secrets.config import SecretsServiceConfig


def test_secrets_service_config_local_key_fields(monkeypatch, tmp_path):
    """Test SecretsServiceConfig allow_key_creation and local_key_creation_path render and default correctly.

    The default key path lives under the NeMo Helix user data directory (XDG-style)
    so it survives macOS ``/tmp/`` cleanup on reboot. We override
    ``NHX_DATA_DIR`` to keep the assertion stable across developer machines.
    """
    monkeypatch.setenv("NHX_DATA_DIR", str(tmp_path))
    config = SecretsServiceConfig()
    assert config.allow_key_creation is False
    assert config.local_key_creation_path == str(nhx_user_data_dir() / "nhx-encryption-key.txt")

    config_with_overrides = SecretsServiceConfig(
        allow_key_creation=True,
        local_key_creation_path="/data/nhx-encryption-key.txt",
    )
    assert config_with_overrides.allow_key_creation is True
    assert config_with_overrides.local_key_creation_path == "/data/nhx-encryption-key.txt"


def test_local_key_creation_raises_when_allow_key_creation_false():
    """Test local_key_creation raises when allow_key_creation is False."""
    config = SecretsServiceConfig(allow_key_creation=False)
    Configuration.set_override(config)
    try:
        with pytest.raises(ValueError, match="allow_key_creation"):
            local_key_creation()
    finally:
        Configuration.clear_override(SecretsServiceConfig)


def test_local_key_creation_creates_key_file_when_missing(tmp_path):
    """Test local_key_creation creates and persists a new key file when path does not exist."""
    key_path = tmp_path / "nhx-encryption-key.txt"
    assert not key_path.exists()

    config = SecretsServiceConfig(
        allow_key_creation=True,
        local_key_creation_path=str(key_path),
    )
    Configuration.set_override(config)
    try:
        encryptor = local_key_creation()
        assert key_path.exists()
        key_content = key_path.read_text().strip()
        assert len(key_content) > 0
        # Encryptor should work with the created key
        plaintext = "test secret"
        assert encryptor.decrypt(encryptor.encrypt(plaintext)) == plaintext
    finally:
        Configuration.clear_override(SecretsServiceConfig)


def test_local_key_creation_uses_existing_key_file(tmp_path):
    """Test local_key_creation uses existing key file and does not overwrite it."""
    key_path = tmp_path / "nhx-encryption-key.txt"
    existing_key = get_base64_encoded_random_bytes(32)
    key_path.write_text(existing_key)

    config = SecretsServiceConfig(
        allow_key_creation=True,
        local_key_creation_path=str(key_path),
    )
    Configuration.set_override(config)
    try:
        encryptor1 = local_key_creation()
        encryptor2 = local_key_creation()
        # Same key file => same encryptor behavior (can decrypt each other's ciphertext)
        plaintext = "shared secret"
        encrypted = encryptor1.encrypt(plaintext)
        assert encryptor2.decrypt(encrypted) == plaintext
        # File was not overwritten
        assert key_path.read_text().strip() == existing_key
    finally:
        Configuration.clear_override(SecretsServiceConfig)


def test_get_encryptor_by_name_empty_returns_local_when_allow_key_creation(tmp_path):
    """Test get_encryptor_by_name('') returns local key encryptor when allow_key_creation is True."""
    key_path = tmp_path / "key.txt"
    key_path.write_text(get_base64_encoded_random_bytes(32))

    config = SecretsServiceConfig(
        allow_key_creation=True,
        local_key_creation_path=str(key_path),
    )
    Configuration.set_override(config)
    try:
        encryptor = get_encryptor_by_name("")
        assert isinstance(encryptor, SecretKeyEncryptor)
        assert encryptor.decrypt(encryptor.encrypt("data")) == "data"
    finally:
        Configuration.clear_override(SecretsServiceConfig)


def test_get_encryptor_by_name_empty_raises_when_allow_key_creation_false():
    """Test get_encryptor_by_name('') raises when allow_key_creation is False and no provider for ''."""
    config = SecretsServiceConfig(allow_key_creation=False)
    Configuration.set_override(config)
    try:
        with pytest.raises(ValueError, match="No encryptor configuration found"):
            get_encryptor_by_name("")
    finally:
        Configuration.clear_override(SecretsServiceConfig)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from cryptography.fernet import Fernet, InvalidToken
from pydantic import ValidationError

try:
    from scaled_evals.api import crypto
    from scaled_evals.api.settings import Settings, settings
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def test_previous_key_decrypts_and_reencrypt_uses_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    primary_key = Fernet.generate_key().decode()
    previous_key = Fernet.generate_key().decode()
    old_ciphertext = Fernet(previous_key.encode()).encrypt(b"secret")

    monkeypatch.setattr(settings, "credentials_encryption_key", primary_key)
    monkeypatch.setattr(settings, "credentials_encryption_key_previous", previous_key)

    assert crypto.decrypt(old_ciphertext) == "secret"
    rotated = crypto.reencrypt(old_ciphertext)
    assert Fernet(primary_key.encode()).decrypt(rotated) == b"secret"
    with pytest.raises(InvalidToken):
        Fernet(previous_key.encode()).decrypt(rotated)


def test_encrypt_uses_only_primary_key(monkeypatch: pytest.MonkeyPatch) -> None:
    primary_key = Fernet.generate_key().decode()
    previous_key = Fernet.generate_key().decode()
    monkeypatch.setattr(settings, "credentials_encryption_key", primary_key)
    monkeypatch.setattr(settings, "credentials_encryption_key_previous", previous_key)

    ciphertext = crypto.encrypt("secret")

    assert Fernet(primary_key.encode()).decrypt(ciphertext) == b"secret"
    with pytest.raises(InvalidToken):
        Fernet(previous_key.encode()).decrypt(ciphertext)


def test_invalid_key_configuration_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "credentials_encryption_key", "not-a-fernet-key")
    monkeypatch.setattr(settings, "credentials_encryption_key_previous", "")

    with pytest.raises(RuntimeError, match="invalid credential encryption key configuration"):
        crypto.encrypt("secret")


def test_multiple_previous_keys_are_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    primary_key = Fernet.generate_key().decode()
    previous_keys = [Fernet.generate_key().decode(), Fernet.generate_key().decode()]
    ciphertext = Fernet(previous_keys[1].encode()).encrypt(b"secret")
    monkeypatch.setattr(settings, "credentials_encryption_key", primary_key)
    monkeypatch.setattr(settings, "credentials_encryption_key_previous", ",".join(previous_keys))

    assert crypto.decrypt(ciphertext) == "secret"


def test_settings_require_a_valid_primary_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CREDENTIALS_ENCRYPTION_KEY")
    with pytest.raises(ValidationError, match="credentials_encryption_key"):
        Settings(_env_file=None)  # ty: ignore[missing-argument, unknown-argument]

    monkeypatch.setenv("CREDENTIALS_ENCRYPTION_KEY", "not-a-fernet-key")
    with pytest.raises(ValidationError, match="must be a valid Fernet key"):
        Settings(_env_file=None)  # ty: ignore[missing-argument, unknown-argument]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from nhx.common.auth import signing_keys
from nhx.common.auth.signing_keys import RSASigningKeyCache


def _private_key_pem() -> bytes:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _write_private_key(path: Path) -> None:
    path.write_bytes(_private_key_pem())


def test_signing_key_cache_reads_private_key_file_once_for_same_file_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_path = tmp_path / "private.pem"
    _write_private_key(key_path)
    original_load = signing_keys._load_rsa_signing_key_async
    load_count = 0

    async def counted_load(**kwargs: Any) -> signing_keys.RSASigningKey:
        nonlocal load_count
        load_count += 1
        return await original_load(**kwargs)

    monkeypatch.setattr(signing_keys, "_load_rsa_signing_key_async", counted_load)
    cache = RSASigningKeyCache()

    first = cache.get_from_file(
        kid="test-key",
        private_key_file=str(key_path),
        missing_private_key_message="private key file is required",
        invalid_private_key_message="private key must be RSA",
    )
    second = cache.get_from_file(
        kid="test-key",
        private_key_file=str(key_path),
        missing_private_key_message="private key file is required",
        invalid_private_key_message="private key must be RSA",
    )

    assert first is second
    assert load_count == 1


def test_signing_key_cache_reloads_when_file_metadata_changes(tmp_path: Path) -> None:
    key_path = tmp_path / "private.pem"
    _write_private_key(key_path)
    cache = RSASigningKeyCache()

    first = cache.get_from_file(
        kid="test-key",
        private_key_file=str(key_path),
        missing_private_key_message="private key file is required",
        invalid_private_key_message="private key must be RSA",
    )
    _write_private_key(key_path)
    stat = key_path.stat()
    os.utime(key_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = cache.get_from_file(
        kid="test-key",
        private_key_file=str(key_path),
        missing_private_key_message="private key file is required",
        invalid_private_key_message="private key must be RSA",
    )

    assert second is not first
    assert second.public_jwk["kid"] == "test-key"
    assert second.public_jwk["n"] != first.public_jwk["n"]


def test_signing_key_cache_reloads_when_kid_changes(tmp_path: Path) -> None:
    key_path = tmp_path / "private.pem"
    _write_private_key(key_path)
    cache = RSASigningKeyCache()

    first = cache.get_from_file(
        kid="first-key",
        private_key_file=str(key_path),
        missing_private_key_message="private key file is required",
        invalid_private_key_message="private key must be RSA",
    )
    second = cache.get_from_file(
        kid="second-key",
        private_key_file=str(key_path),
        missing_private_key_message="private key file is required",
        invalid_private_key_message="private key must be RSA",
    )

    assert second is not first
    assert first.public_jwk["kid"] == "first-key"
    assert second.public_jwk["kid"] == "second-key"


def test_public_jwk_from_file_returns_copy(tmp_path: Path) -> None:
    key_path = tmp_path / "private.pem"
    _write_private_key(key_path)
    cache = RSASigningKeyCache()

    first = cache.public_jwk_from_file(
        kid="test-key",
        private_key_file=str(key_path),
        missing_private_key_message="private key file is required",
        invalid_private_key_message="private key must be RSA",
    )
    first["kid"] = "mutated"
    second = cache.public_jwk_from_file(
        kid="test-key",
        private_key_file=str(key_path),
        missing_private_key_message="private key file is required",
        invalid_private_key_message="private key must be RSA",
    )

    assert second["kid"] == "test-key"
    assert second["use"] == "sig"
    assert second["alg"] == "RS256"


def test_signing_key_cache_rejects_missing_private_key_file() -> None:
    cache = RSASigningKeyCache()

    with pytest.raises(RuntimeError, match="private key file is required"):
        cache.get_from_file(
            kid="test-key",
            private_key_file=None,
            missing_private_key_message="private key file is required",
            invalid_private_key_message="private key must be RSA",
        )


def test_signing_key_cache_maps_missing_private_key_path_to_missing_message(tmp_path: Path) -> None:
    cache = RSASigningKeyCache()

    with pytest.raises(RuntimeError, match="private key file is required"):
        cache.get_from_file(
            kid="test-key",
            private_key_file=str(tmp_path / "missing.pem"),
            missing_private_key_message="private key file is required",
            invalid_private_key_message="private key must be RSA",
        )


def test_signing_key_cache_maps_unreadable_private_key_file_to_missing_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_path = tmp_path / "private.pem"
    _write_private_key(key_path)

    class UnreadableFile:
        async def __aenter__(self) -> None:
            raise PermissionError("permission denied")

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

    def unreadable_open(*args: object, **kwargs: object) -> UnreadableFile:
        return UnreadableFile()

    monkeypatch.setattr(signing_keys.aiofiles, "open", unreadable_open)
    cache = RSASigningKeyCache()

    with pytest.raises(RuntimeError, match="private key file is required"):
        cache.get_from_file(
            kid="test-key",
            private_key_file=str(key_path),
            missing_private_key_message="private key file is required",
            invalid_private_key_message="private key must be RSA",
        )


def test_signing_key_cache_maps_malformed_private_key_to_invalid_message(tmp_path: Path) -> None:
    key_path = tmp_path / "private.pem"
    key_path.write_text("not a pem")
    cache = RSASigningKeyCache()

    with pytest.raises(RuntimeError, match="private key must be RSA"):
        cache.get_from_file(
            kid="test-key",
            private_key_file=str(key_path),
            missing_private_key_message="private key file is required",
            invalid_private_key_message="private key must be RSA",
        )


def test_signing_key_cache_maps_encrypted_private_key_to_invalid_message(tmp_path: Path) -> None:
    key_path = tmp_path / "private.pem"
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(b"password"),
        )
    )
    cache = RSASigningKeyCache()

    with pytest.raises(RuntimeError, match="private key must be RSA"):
        cache.get_from_file(
            kid="test-key",
            private_key_file=str(key_path),
            missing_private_key_message="private key file is required",
            invalid_private_key_message="private key must be RSA",
        )


def test_signing_key_cache_rejects_non_rsa_key(tmp_path: Path) -> None:
    from cryptography.hazmat.primitives.asymmetric import ed25519

    key_path = tmp_path / "private.pem"
    private_key = ed25519.Ed25519PrivateKey.generate()
    key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cache = RSASigningKeyCache()

    with pytest.raises(RuntimeError, match="private key must be RSA"):
        cache.get_from_file(
            kid="test-key",
            private_key_file=str(key_path),
            missing_private_key_message="private key file is required",
            invalid_private_key_message="private key must be RSA",
        )


def test_async_signing_key_cache_reads_private_key_file_once_for_same_file_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_path = tmp_path / "private.pem"
    _write_private_key(key_path)
    original_load = signing_keys._load_rsa_signing_key_async
    load_count = 0

    async def counted_load(**kwargs: Any) -> signing_keys.RSASigningKey:
        nonlocal load_count
        load_count += 1
        return await original_load(**kwargs)

    monkeypatch.setattr(signing_keys, "_load_rsa_signing_key_async", counted_load)
    cache = RSASigningKeyCache()

    async_key = asyncio.run(
        cache.get_from_file_async(
            kid="test-key",
            private_key_file=str(key_path),
            missing_private_key_message="private key file is required",
            invalid_private_key_message="private key must be RSA",
        )
    )
    second_async_key = asyncio.run(
        cache.get_from_file_async(
            kid="test-key",
            private_key_file=str(key_path),
            missing_private_key_message="private key file is required",
            invalid_private_key_message="private key must be RSA",
        )
    )

    assert second_async_key is async_key
    assert load_count == 1


def test_signing_key_cache_sync_and_async_use_same_cached_entry(tmp_path: Path) -> None:
    key_path = tmp_path / "private.pem"
    _write_private_key(key_path)
    cache = RSASigningKeyCache()

    sync_key = cache.get_from_file(
        kid="test-key",
        private_key_file=str(key_path),
        missing_private_key_message="private key file is required",
        invalid_private_key_message="private key must be RSA",
    )
    async_key = asyncio.run(
        cache.get_from_file_async(
            kid="test-key",
            private_key_file=str(key_path),
            missing_private_key_message="private key file is required",
            invalid_private_key_message="private key must be RSA",
        )
    )

    assert async_key is sync_key


async def test_signing_key_cache_sync_wrapper_rejects_running_event_loop(tmp_path: Path) -> None:
    key_path = tmp_path / "private.pem"
    _write_private_key(key_path)
    cache = RSASigningKeyCache()

    with pytest.raises(RuntimeError, match="Use RSASigningKeyCache async methods"):
        cache.get_from_file(
            kid="test-key",
            private_key_file=str(key_path),
            missing_private_key_message="private key file is required",
            invalid_private_key_message="private key must be RSA",
        )


def test_concurrent_async_signing_key_requests_initialize_cache_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_path = tmp_path / "private.pem"
    _write_private_key(key_path)
    original_load = signing_keys._load_rsa_signing_key_async
    load_count = 0

    async def counted_load(**kwargs: Any) -> signing_keys.RSASigningKey:
        nonlocal load_count
        load_count += 1
        return await original_load(**kwargs)

    monkeypatch.setattr(signing_keys, "_load_rsa_signing_key_async", counted_load)
    cache = RSASigningKeyCache()

    async def load_many() -> list[object]:
        return await asyncio.gather(
            *[
                cache.get_from_file_async(
                    kid="test-key",
                    private_key_file=str(key_path),
                    missing_private_key_message="private key file is required",
                    invalid_private_key_message="private key must be RSA",
                )
                for _ in range(10)
            ]
        )

    keys = asyncio.run(load_many())

    assert all(key is keys[0] for key in keys)
    assert load_count == 1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import aiofiles
import aiofiles.os
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from .loading_cache import AsyncLoadingCache

ValueT = TypeVar("ValueT")


@dataclass(frozen=True)
class RSASigningKey:
    """Parsed RSA signing key material plus its public JWKS representation."""

    private_key: rsa.RSAPrivateKey
    public_key: rsa.RSAPublicKey
    kid: str
    public_jwk: dict[str, Any]


@dataclass(frozen=True)
class _FileCacheKey:
    kid: str
    private_key_file: str
    mtime_ns: int
    size: int


class RSASigningKeyCache:
    """Cache RSA signing keys loaded from PEM files."""

    def __init__(self) -> None:
        self._cache: AsyncLoadingCache[_FileCacheKey, RSASigningKey] = AsyncLoadingCache()
        self._sync_values: dict[_FileCacheKey, RSASigningKey] = {}

    def clear(self) -> None:
        self._sync_values.clear()
        _run_async(self.clear_async)

    async def clear_async(self) -> None:
        self._sync_values.clear()
        await self._cache.clear()

    def get_from_file(
        self,
        *,
        kid: str,
        private_key_file: str | None,
        missing_private_key_message: str,
        invalid_private_key_message: str,
    ) -> RSASigningKey:
        _reject_running_loop_for_sync_api()
        if not kid:
            raise RuntimeError("token signing key id must be configured")
        if not private_key_file:
            raise RuntimeError(missing_private_key_message)

        path = Path(private_key_file)
        try:
            cache_key = _file_cache_key(kid, path)
        except OSError as exc:
            raise RuntimeError(missing_private_key_message) from exc

        signing_key = self._sync_values.get(cache_key)
        if signing_key is None:
            signing_key = _load_rsa_signing_key(
                kid=kid,
                path=path,
                missing_private_key_message=missing_private_key_message,
                invalid_private_key_message=invalid_private_key_message,
            )
            self._sync_values[cache_key] = signing_key
        return signing_key

    async def get_from_file_async(
        self,
        *,
        kid: str,
        private_key_file: str | None,
        missing_private_key_message: str,
        invalid_private_key_message: str,
    ) -> RSASigningKey:
        if not kid:
            raise RuntimeError("token signing key id must be configured")
        if not private_key_file:
            raise RuntimeError(missing_private_key_message)

        path = Path(private_key_file)
        try:
            cache_key = await _file_cache_key_async(kid, path)
        except OSError as exc:
            raise RuntimeError(missing_private_key_message) from exc
        return await self._cache.get_or_load(
            cache_key,
            lambda: _load_rsa_signing_key_async(
                kid=kid,
                path=path,
                missing_private_key_message=missing_private_key_message,
                invalid_private_key_message=invalid_private_key_message,
            ),
        )

    def public_jwk_from_file(
        self,
        *,
        kid: str,
        private_key_file: str | None,
        missing_private_key_message: str,
        invalid_private_key_message: str,
    ) -> dict[str, Any]:
        signing_key = self.get_from_file(
            kid=kid,
            private_key_file=private_key_file,
            missing_private_key_message=missing_private_key_message,
            invalid_private_key_message=invalid_private_key_message,
        )
        return dict(signing_key.public_jwk)

    async def public_jwk_from_file_async(
        self,
        *,
        kid: str,
        private_key_file: str | None,
        missing_private_key_message: str,
        invalid_private_key_message: str,
    ) -> dict[str, Any]:
        signing_key = await self.get_from_file_async(
            kid=kid,
            private_key_file=private_key_file,
            missing_private_key_message=missing_private_key_message,
            invalid_private_key_message=invalid_private_key_message,
        )
        return dict(signing_key.public_jwk)


def _run_async(factory: Callable[[], Coroutine[Any, Any, ValueT]]) -> ValueT:
    _reject_running_loop_for_sync_api()
    return asyncio.run(factory())


def _reject_running_loop_for_sync_api() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError("Use RSASigningKeyCache async methods from async contexts")


async def _load_rsa_signing_key_async(
    *,
    kid: str,
    path: Path,
    missing_private_key_message: str,
    invalid_private_key_message: str,
) -> RSASigningKey:
    try:
        async with aiofiles.open(path, "rb") as key_file:
            private_key_pem = await key_file.read()
    except OSError as exc:
        raise RuntimeError(missing_private_key_message) from exc

    return _rsa_signing_key_from_pem(
        kid=kid,
        private_key_pem=private_key_pem,
        invalid_private_key_message=invalid_private_key_message,
    )


def _load_rsa_signing_key(
    *,
    kid: str,
    path: Path,
    missing_private_key_message: str,
    invalid_private_key_message: str,
) -> RSASigningKey:
    try:
        private_key_pem = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(missing_private_key_message) from exc

    return _rsa_signing_key_from_pem(
        kid=kid,
        private_key_pem=private_key_pem,
        invalid_private_key_message=invalid_private_key_message,
    )


async def _file_cache_key_async(kid: str, path: Path) -> _FileCacheKey:
    stat = await aiofiles.os.stat(path)
    return _FileCacheKey(
        kid=kid,
        private_key_file=str(path.expanduser().resolve(strict=False)),
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
    )


def _file_cache_key(kid: str, path: Path) -> _FileCacheKey:
    stat = path.stat()
    return _FileCacheKey(
        kid=kid,
        private_key_file=str(path.expanduser().resolve(strict=False)),
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
    )


def _rsa_signing_key_from_pem(
    *,
    kid: str,
    private_key_pem: bytes,
    invalid_private_key_message: str,
) -> RSASigningKey:
    try:
        private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
        raise RuntimeError(invalid_private_key_message) from exc
    if not isinstance(private_key, rsa.RSAPrivateKey):
        raise RuntimeError(invalid_private_key_message)

    public_key = private_key.public_key()
    jwk = json.loads(RSAAlgorithm.to_jwk(public_key))
    jwk.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return RSASigningKey(
        private_key=private_key,
        public_key=public_key,
        kid=kid,
        public_jwk=jwk,
    )

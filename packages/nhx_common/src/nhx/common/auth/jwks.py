# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async JWKS fetching and signing-key resolution for bearer-token validation."""

from __future__ import annotations

import time
from typing import Any

import jwt
from nhx.common import http_clients

from .loading_cache import AsyncCoalescingLoader

DEFAULT_JWKS_CACHE_LIFESPAN = 3600
UNKNOWN_KID_REFRESH_MIN_INTERVAL_SECONDS = 5.0


class _UnknownJWKKeyIDError(jwt.InvalidTokenError):
    def __init__(self, key_id: str) -> None:
        self.key_id = key_id
        super().__init__(f'Unable to find a signing key that matches: "{key_id}"')


def validate_jwks(jwks: dict[str, Any]) -> None:
    """Raise when a JWKS document cannot be parsed by PyJWT."""
    jwt.PyJWKSet.from_dict(jwks)


def signing_jwk_from_jwks(token: str, jwks: dict[str, Any]) -> Any:
    """Resolve the JWK matching the token's ``kid`` from a JWKS document."""
    key_id = jwt.get_unverified_header(token).get("kid")
    if not isinstance(key_id, str) or not key_id:
        raise jwt.InvalidTokenError("JWT did not include a signing key id")

    jwk_set = jwt.PyJWKSet.from_dict(jwks)
    for jwk in jwk_set.keys:
        if jwk.public_key_use in ("sig", None) and jwk.key_id == key_id:
            return jwk
    raise _UnknownJWKKeyIDError(key_id)


# Keep JWKS retrieval async. PyJWT's PyJWKClient performs synchronous network I/O
# and can block async request handlers.
class AsyncJWKSClient:
    """Async JWKS client with TTL caching and one refresh on unknown key IDs."""

    def __init__(self, jwks_uri: str, *, lifespan: int = DEFAULT_JWKS_CACHE_LIFESPAN) -> None:
        self._jwks_uri = jwks_uri
        self._lifespan = lifespan
        self._jwks: dict[str, Any] | None = None
        self._jwks_cache_time = 0.0
        self._unknown_kid_refresh_loader: AsyncCoalescingLoader[dict[str, Any]] = AsyncCoalescingLoader(
            min_interval_seconds=UNKNOWN_KID_REFRESH_MIN_INTERVAL_SECONDS
        )

    async def get_signing_key_from_jwt(self, token: str) -> Any:
        jwks, cache_hit = await self._fetch_jwks()
        try:
            return signing_jwk_from_jwks(token, jwks)
        except _UnknownJWKKeyIDError:
            if not cache_hit:
                raise
            refreshed_jwks = await self._refresh_jwks_for_unknown_kid()
            return signing_jwk_from_jwks(token, refreshed_jwks)

    async def get_jwks(self) -> dict[str, Any]:
        jwks, _ = await self._fetch_jwks()
        return jwks

    def clear_cache(self) -> None:
        self._jwks = None
        self._jwks_cache_time = 0.0
        self._unknown_kid_refresh_loader = AsyncCoalescingLoader(
            min_interval_seconds=UNKNOWN_KID_REFRESH_MIN_INTERVAL_SECONDS
        )

    async def _refresh_jwks_for_unknown_kid(self) -> dict[str, Any]:
        return await self._unknown_kid_refresh_loader.load(
            self._force_refresh_jwks,
            rate_limited_value=self._cached_jwks,
        )

    def _cached_jwks(self) -> dict[str, Any]:
        if self._jwks is None:
            raise jwt.InvalidTokenError("JWKS cache is empty")
        return self._jwks

    async def _force_refresh_jwks(self) -> dict[str, Any]:
        jwks, _ = await self._fetch_jwks(refresh=True)
        return jwks

    async def _fetch_jwks(self, *, refresh: bool = False) -> tuple[dict[str, Any], bool]:
        now = time.monotonic()
        if self._jwks is not None and not refresh and self._lifespan > 0:
            if now - self._jwks_cache_time < self._lifespan:
                return self._jwks, True

        response = await http_clients.shared_async_http_client().get(self._jwks_uri, timeout=10.0)
        response.raise_for_status()
        jwks = response.json()
        if not isinstance(jwks, dict):
            raise jwt.InvalidTokenError("JWKS response was not an object")
        validate_jwks(jwks)
        if self._lifespan > 0:
            self._jwks = jwks
            self._jwks_cache_time = now
        return jwks, False

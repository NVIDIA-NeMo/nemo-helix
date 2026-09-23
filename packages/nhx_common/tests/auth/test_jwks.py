# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from nhx.common import http_clients
from nhx.common.auth.jwks import AsyncJWKSClient

JWKS_URI = "https://auth.example.test/jwks"


def _rsa_key_and_jwk(kid: str) -> tuple[Any, dict[str, Any]]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = RSAAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    jwk.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return private_key, jwk


def _token(private_key: Any, *, kid: str | None) -> str:
    headers = {"kid": kid} if kid is not None else None
    return jwt.encode({"sub": "user"}, private_key, algorithm="RS256", headers=headers)


class FakeResponse:
    def __init__(self, jwks: dict[str, Any]) -> None:
        self._jwks = jwks

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._jwks


class FakeAsyncClient:
    def __init__(self, jwks: dict[str, Any]) -> None:
        self._jwks = jwks
        self.calls = 0

    async def get(self, url: str, *, timeout: float) -> FakeResponse:
        assert url == JWKS_URI
        assert timeout == 10.0
        self.calls += 1
        await asyncio.sleep(0.01)
        return FakeResponse(self._jwks)


class SequenceFakeAsyncClient:
    def __init__(self, jwks_responses: list[dict[str, Any]]) -> None:
        self._jwks_responses = jwks_responses
        self.calls = 0

    async def get(self, url: str, *, timeout: float) -> FakeResponse:
        assert url == JWKS_URI
        assert timeout == 10.0
        self.calls += 1
        await asyncio.sleep(0.01)
        if len(self._jwks_responses) > 1:
            return FakeResponse(self._jwks_responses.pop(0))
        return FakeResponse(self._jwks_responses[0])


@pytest.mark.parametrize("bad_token", ["malformed-token", None])
async def test_cached_jwks_lookup_does_not_refresh_malformed_or_missing_kid_tokens(monkeypatch, bad_token):
    private_key, jwk = _rsa_key_and_jwk("known-key")
    fake_client = FakeAsyncClient({"keys": [jwk]})
    monkeypatch.setattr(http_clients, "shared_async_http_client", lambda: fake_client)
    client = AsyncJWKSClient(JWKS_URI)
    await client.get_signing_key_from_jwt(_token(private_key, kid="known-key"))

    token = bad_token if bad_token is not None else _token(private_key, kid=None)
    with pytest.raises(jwt.InvalidTokenError):
        await client.get_signing_key_from_jwt(token)

    assert fake_client.calls == 1


async def test_cached_unknown_kid_concurrent_refresh_uses_single_refreshed_jwks(monkeypatch):
    known_private_key, known_jwk = _rsa_key_and_jwk("known-key")
    rotated_private_key, rotated_jwk = _rsa_key_and_jwk("rotated-key")
    fake_client = SequenceFakeAsyncClient([{"keys": [known_jwk]}, {"keys": [rotated_jwk]}])
    monkeypatch.setattr(http_clients, "shared_async_http_client", lambda: fake_client)
    client = AsyncJWKSClient(JWKS_URI)
    await client.get_signing_key_from_jwt(_token(known_private_key, kid="known-key"))
    rotated_token = _token(rotated_private_key, kid="rotated-key")

    results = await asyncio.gather(*(client.get_signing_key_from_jwt(rotated_token) for _ in range(5)))

    assert [result.key_id for result in results] == ["rotated-key"] * 5
    assert fake_client.calls == 2


async def test_cached_unknown_kid_refresh_is_rate_limited(monkeypatch):
    private_key, jwk = _rsa_key_and_jwk("known-key")
    fake_client = FakeAsyncClient({"keys": [jwk]})
    monkeypatch.setattr(http_clients, "shared_async_http_client", lambda: fake_client)
    client = AsyncJWKSClient(JWKS_URI)
    await client.get_signing_key_from_jwt(_token(private_key, kid="known-key"))
    unknown_token = _token(private_key, kid="unknown-key")

    results = await asyncio.gather(
        *(client.get_signing_key_from_jwt(unknown_token) for _ in range(5)),
        return_exceptions=True,
    )

    assert all(isinstance(result, jwt.InvalidTokenError) for result in results)
    assert fake_client.calls == 2

    with pytest.raises(jwt.InvalidTokenError):
        await client.get_signing_key_from_jwt(unknown_token)

    assert fake_client.calls == 2

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugins trust an access key's claims only when the platform signed the key."""

from __future__ import annotations

import base64
import builtins
import json
import time
from pathlib import Path
from typing import Any

import jwt
import nhx.common.auth.access_keys as access_keys_mod
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from nemo_helix_plugin.auth import verified_access_key_claims
from nemo_helix_plugin.auth.access_keys.types import AccessKeyCreateRequest
from nemo_helix_plugin.client.oidc import generate_unsigned_jwt
from nhx.common.auth.access_keys import AccessKeyIssuerService, public_jwk_from_private_key_pem_async
from nhx.common.auth.jwks import signing_jwk_from_jwks
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.common.config.base import AccessKeyConfig, TokenSigningConfig

ATTRIBUTES = {"evals": {"kind": "trial", "session": "run-1:hello__abc"}}


def _pem() -> bytes:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _config(tmp_path: Path, name: str = "platform") -> AuthConfig:
    key_path = tmp_path / f"{name}.pem"
    key_path.write_bytes(_pem())
    return AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(
            issuer="https://nhx.example.test/apis/auth", key_id="platform-key", private_key_file=str(key_path)
        ),
        access_keys=AccessKeyConfig(enabled=True, audience="nemo-helix-access-key"),
    )


async def _key(config: AuthConfig, *, minted_at: float | None = None) -> str:
    issuer = AccessKeyIssuerService(
        config=config,
        principal=Principal(id="alice@example.com"),
        acting_service="evals",
        now=lambda: int(minted_at if minted_at is not None else time.time()),
    )
    request = AccessKeyCreateRequest(name="trial", expires_in_seconds=600, service_attributes=ATTRIBUTES["evals"])
    return (await issuer.create_async(request)).token


@pytest.fixture
async def platform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AuthConfig:
    """The platform's auth config, serving its own public key as its JWKS."""
    config = _config(tmp_path)
    jwks = {"keys": [await public_jwk_from_private_key_pem_async(config)]}

    async def from_jwks(_config: AuthConfig, token: str) -> Any:
        return signing_jwk_from_jwks(token, jwks).key

    monkeypatch.setattr("nhx.common.config.get_auth_config", lambda: config)
    monkeypatch.setattr(access_keys_mod, "_access_key_signing_key_from_remote_jwks", from_jwks)
    return config


async def test_a_key_the_platform_signed_yields_its_service_attributes(platform: AuthConfig) -> None:
    claims = await verified_access_key_claims(await _key(platform))

    assert claims is not None
    assert claims["sub"] == "alice@example.com"
    assert claims["nhx_service_attributes"] == ATTRIBUTES


async def test_an_unsigned_token_is_refused_whatever_it_claims(platform: AuthConfig) -> None:
    """The gateway may accept unsigned tokens (``allow_unsigned_jwt``), so readers mustn't take its word."""
    signed = jwt.decode(await _key(platform), options={"verify_signature": False})
    unsigned = generate_unsigned_jwt(signed["sub"], extra_claims={k: v for k, v in signed.items() if k != "sub"})

    assert await verified_access_key_claims(unsigned) is None


async def test_a_key_signed_by_another_key_is_refused(platform: AuthConfig, tmp_path: Path) -> None:
    impostor = _config(tmp_path, name="impostor")

    assert await verified_access_key_claims(await _key(impostor)) is None


async def test_a_key_with_an_altered_claim_is_refused(platform: AuthConfig) -> None:
    header, payload, signature = (await _key(platform)).split(".")
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    claims["nhx_service_attributes"]["evals"]["session"] = "run-1:someone-else"
    altered = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")

    assert await verified_access_key_claims(f"{header}.{altered}.{signature}") is None


async def test_an_expired_key_is_refused(platform: AuthConfig) -> None:
    token = await _key(platform, minted_at=time.time() - 3600)

    assert await verified_access_key_claims(token) is None


async def test_nothing_is_trusted_outside_the_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def without_nhx_common(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("nhx.common"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_nhx_common)

    assert await verified_access_key_claims("a.b.c") is None

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from unittest.mock import patch

import httpx
import jwt
import pytest
import pytest_asyncio
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from nhx.common import http_clients
from nhx.common.auth.discovery import BearerTokenSource
from nhx.common.auth.jwt import JWTValidator
from nhx.common.auth.oidc_bearer import OIDCBearerPurpose, id_token_bearer_allowed, oidc_bearer_purpose
from nhx.common.config import AuthConfig
from nhx.common.config.base import (
    OIDCConfidentialClientConfig,
    OIDCConfig,
    OIDCPublicClientConfig,
    OIDCServerSessionsConfig,
)


@pytest.mark.parametrize(
    ("header", "claims", "expected"),
    [
        ({"typ": "at+jwt"}, {}, OIDCBearerPurpose.ACCESS_TOKEN),
        ({"typ": "application/at+jwt"}, {}, OIDCBearerPurpose.ACCESS_TOKEN),
        ({"typ": "JWT"}, {"typ": "Bearer"}, OIDCBearerPurpose.ACCESS_TOKEN),
        ({"typ": "JWT"}, {"typ": "ID"}, OIDCBearerPurpose.ID_TOKEN),
        ({}, {"token_use": "access"}, OIDCBearerPurpose.ACCESS_TOKEN),
        ({}, {"token_use": "id"}, OIDCBearerPurpose.ID_TOKEN),
        ({"typ": "JWT"}, {"scope": "platform:read"}, OIDCBearerPurpose.UNSPECIFIED),
        ({"typ": "at+jwt"}, {"typ": "ID"}, OIDCBearerPurpose.INVALID),
        ({}, {"typ": "Bearer", "token_use": "id"}, OIDCBearerPurpose.INVALID),
        ({"typ": "secevent+jwt"}, {}, OIDCBearerPurpose.INVALID),
        ({}, {"typ": "Refresh"}, OIDCBearerPurpose.INVALID),
        ({}, {"token_use": "refresh"}, OIDCBearerPurpose.INVALID),
        ({"typ": None}, {}, OIDCBearerPurpose.INVALID),
        ({}, {"typ": []}, OIDCBearerPurpose.INVALID),
    ],
)
def test_signed_purpose_signals_are_consistent(
    header: dict[str, object], claims: dict[str, object], expected: OIDCBearerPurpose
) -> None:
    assert oidc_bearer_purpose(header, claims) is expected


@pytest.fixture(scope="module")
def signing_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@dataclass
class SignedProvider:
    signing_key: rsa.RSAPrivateKey
    config: AuthConfig
    userinfo_tokens: set[str] = field(default_factory=set)
    requests: list[str] = field(default_factory=list)
    userinfo_subject: str = "user-1"

    def token(self, claims: dict[str, object], header: dict[str, str] | None = None) -> str:
        now = int(time.time())
        return jwt.encode(
            {"iss": "https://issuer.example", "sub": "user-1", "iat": now, "exp": now + 600, **claims},
            self.signing_key,
            algorithm="RS256",
            headers={"kid": "test-key", **(header or {})},
        )

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request.url.path)
        if request.url.path == "/jwks":
            public_key = RSAAlgorithm.to_jwk(self.signing_key.public_key(), as_dict=True)
            return httpx.Response(200, json={"keys": [{**public_key, "kid": "test-key", "alg": "RS256"}]})
        if request.url.path == "/userinfo":
            token = request.headers.get("Authorization", "").removeprefix("Bearer ")
            if token in self.userinfo_tokens:
                return httpx.Response(200, json={"sub": self.userinfo_subject})
            return httpx.Response(401)
        raise AssertionError(f"Unexpected provider request: {request.url.path}")


@pytest_asyncio.fixture
async def provider(signing_key: rsa.RSAPrivateKey) -> AsyncIterator[SignedProvider]:
    provider = SignedProvider(
        signing_key,
        AuthConfig(
            enabled=True,
            oidc=OIDCConfig(
                enabled=True,
                issuer="https://issuer.example",
                jwks_uri="https://issuer.example/jwks",
                userinfo_endpoint="https://issuer.example/userinfo",
                public_client=OIDCPublicClientConfig(client_id="public-client"),
                server_sessions=OIDCServerSessionsConfig(encryption_key_env_var="TEST_SESSION_KEY"),
            ),
        ),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider.respond)) as client:
        with patch.object(http_clients, "shared_async_http_client", return_value=client):
            yield provider


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["access_token", "id_token"])
@pytest.mark.parametrize("audience", [None, "public-client"])
@pytest.mark.parametrize("format", ["keycloak", "cognito", "rfc9068", "generic"])
async def test_signed_pair_follows_configured_client_policy(
    provider: SignedProvider, source: BearerTokenSource, audience: str | None, format: str
) -> None:
    provider.config.oidc.public_client = OIDCPublicClientConfig(client_id="public-client", bearer_token_source=source)
    provider.config.oidc.audience = audience
    access_claims: dict[str, object] = {"aud": "public-client", "azp": "public-client", "scope": "openid platform:read"}
    id_claims: dict[str, object] = {"aud": "public-client", "azp": "public-client"}
    access_header: dict[str, str] = {}
    if format == "keycloak":
        access_claims["typ"] = "Bearer"
        id_claims["typ"] = "ID"
    elif format == "cognito":
        access_claims["token_use"] = "access"
        id_claims["token_use"] = "id"
    elif format == "rfc9068":
        access_header["typ"] = "at+jwt"
    access_token = provider.token(access_claims, access_header)
    id_token = provider.token(id_claims)
    provider.userinfo_tokens.add(access_token)
    validator = JWTValidator(provider.config)

    access_identity = await validator.validate_token(access_token)
    id_identity = await validator.validate_token(id_token)

    assert access_identity is not None
    assert access_identity.subject == "user-1"
    assert access_identity.scopes == ["openid", "platform:read"]
    if source == "access_token":
        assert id_identity is None
    else:
        assert id_identity is not None
        assert id_identity.subject == access_identity.subject
        assert id_identity.scopes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("id_profile", ["public", "confidential"])
@pytest.mark.parametrize("typed", [True, False])
async def test_id_token_opt_in_is_bound_to_its_client(provider: SignedProvider, id_profile: str, typed: bool) -> None:
    provider.config.oidc.public_client = OIDCPublicClientConfig(
        client_id="public-client", bearer_token_source="id_token" if id_profile == "public" else "access_token"
    )
    provider.config.oidc.confidential_client = OIDCConfidentialClientConfig(
        client_id="confidential-client",
        client_secret_env_var="TEST_CLIENT_SECRET",
        login_redirect_uri="https://helix.example/callback",
        bearer_token_source="id_token" if id_profile == "confidential" else "access_token",
    )
    validator = JWTValidator(provider.config)
    for client in ("public-client", "confidential-client", "unknown-client"):
        token = provider.token({"aud": client, "azp": client, **({"typ": "ID"} if typed else {})})
        identity = await validator.validate_token(token)
        assert (identity is not None) == (client == f"{id_profile}-client")


@pytest.mark.asyncio
@pytest.mark.parametrize("userinfo_subject", ["user-1", "different-user"])
async def test_untyped_jwt_confirmation_preserves_signed_scopes(
    provider: SignedProvider, userinfo_subject: str
) -> None:
    provider.userinfo_subject = userinfo_subject
    token = provider.token({"aud": "public-client", "scope": "platform:read", "groups": ["editors"]})
    provider.userinfo_tokens.add(token)

    identity = await JWTValidator(provider.config).validate_token(token)

    assert "/userinfo" in provider.requests
    if userinfo_subject == "user-1":
        assert identity is not None
        assert identity.scopes == ["platform:read"]
        assert identity.groups == ["editors"]
    else:
        assert identity is None


@pytest.mark.asyncio
async def test_api_audience_can_distinguish_untyped_access_token(provider: SignedProvider) -> None:
    provider.config.oidc.audience = "helix-api"
    token = provider.token({"aud": "helix-api", "scope": "platform:read", "client_id": "service-client"})

    identity = await JWTValidator(provider.config).validate_token(token)

    assert identity is not None
    assert "/userinfo" not in provider.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("claims", [{"typ": "ID"}, {"token_use": "id"}, {"typ": "Refresh"}])
async def test_wrong_purpose_cannot_fall_back_to_other_provider_endpoints(
    provider: SignedProvider, claims: dict[str, object]
) -> None:
    provider.config.oidc.introspect_opaque_tokens = True
    provider.config.oidc.introspection_endpoint = "https://issuer.example/introspect"
    provider.config.oidc.resolve_opaque_tokens_via_userinfo = True
    token = provider.token({"aud": "public-client", **claims})
    provider.userinfo_tokens.add(token)

    assert await JWTValidator(provider.config).validate_token(token) is None
    assert provider.requests == ["/jwks"]


@pytest.mark.parametrize(
    "claims",
    [
        {},
        {"aud": "unknown"},
        {"aud": ["public-client", "other"]},
        {"aud": "public-client", "azp": "other"},
        {"aud": "public-client", "azp": "public-client", "client_id": "other"},
        {"aud": "public-client", "azp": []},
        {"aud": ["public-client", 42]},
    ],
)
def test_id_token_requires_unambiguous_client_binding(claims: dict[str, object]) -> None:
    config = OIDCConfig(public_client=OIDCPublicClientConfig(client_id="public-client", bearer_token_source="id_token"))
    assert not id_token_bearer_allowed(config, claims)


@pytest.mark.asyncio
async def test_extra_audience_cannot_turn_untyped_id_token_into_api_access(provider: SignedProvider) -> None:
    provider.config.oidc.audience = "helix-api"
    token = provider.token({"aud": ["unknown-client", "helix-api"]})

    assert await JWTValidator(provider.config).validate_token(token) is None
    assert "/userinfo" in provider.requests


@pytest.mark.asyncio
async def test_explicit_id_token_mode_preserves_scopes_carried_by_that_token(provider: SignedProvider) -> None:
    provider.config.oidc.public_client = OIDCPublicClientConfig(
        client_id="public-client", bearer_token_source="id_token"
    )
    token = provider.token({"aud": "public-client", "typ": "ID", "scope": "platform:read"})

    identity = await JWTValidator(provider.config).validate_token(token)

    assert identity is not None
    assert identity.scopes == ["platform:read"]
    assert "/userinfo" not in provider.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["signature", "jwks"])
async def test_failed_jwt_validation_does_not_use_opaque_fallback(provider: SignedProvider, failure: str) -> None:
    provider.config.oidc.resolve_opaque_tokens_via_userinfo = True
    token = provider.token({"aud": "public-client", "typ": "ID"})
    provider.userinfo_tokens.add(token)
    if failure == "signature":
        provider.signing_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        assert await JWTValidator(provider.config).validate_token(token) is None
    else:
        with patch.object(JWTValidator, "_get_jwks_client", side_effect=httpx.ConnectError("unavailable")):
            assert await JWTValidator(provider.config).validate_token(token) is None
    assert "/userinfo" not in provider.requests

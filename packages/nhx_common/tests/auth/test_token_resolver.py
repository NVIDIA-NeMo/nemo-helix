# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nhx.common.auth.token_claims import ActorClaims, TokenClaims
from nhx.common.auth.token_resolver import ResolvedBearerToken, ResolvedTokenKind, resolve_bearer_token
from nhx.common.config import AuthConfig
from nhx.common.config.base import AccessKeyConfig, OIDCConfig


def _claims(subject: str = "alice@example.com") -> TokenClaims:
    return TokenClaims(
        subject=subject,
        email=subject,
        groups=["team-ml"],
        scopes=["models:read"],
        raw_claims={"jti": "ak_example"},
    )


def _claims_with_actor() -> TokenClaims:
    return TokenClaims(
        subject="user:alice",
        email="alice@example.test",
        groups=["researchers"],
        scopes=["models.read"],
        raw_claims={
            "sub": "user:alice",
            "email": "alice@example.test",
            "groups": "researchers",
            "act": {"sub": "service:jobs", "groups": "system:serviceaccounts"},
        },
        actor=ActorClaims(subject="service:jobs", groups=["system:serviceaccounts"]),
    )


def test_resolved_token_maps_actor_claims_to_delegated_principal_headers() -> None:
    claims = _claims_with_actor()
    resolved = ResolvedBearerToken(claims=claims, token_kind="workload_access_token")

    assert resolved.principal.id == "service:jobs"
    assert resolved.principal.groups == ["system:serviceaccounts"]
    assert resolved.principal.on_behalf_of == "user:alice"
    assert resolved.principal.on_behalf_of_email == "alice@example.test"
    assert resolved.principal.on_behalf_of_groups == ["researchers"]
    assert resolved.principal.effective_principal.id == "user:alice"
    assert resolved.principal_headers() == {
        "X-NHX-Principal-Id": "service:jobs",
        "X-NHX-Principal-Groups": "system:serviceaccounts",
        "X-NHX-Actor-Aliases": "service:jobs",
        "X-NHX-Principal-On-Behalf-Of": "user:alice",
        "X-NHX-Principal-On-Behalf-Of-Groups": "researchers",
        "X-NHX-Principal-On-Behalf-Of-Email": "alice@example.test",
        "X-NHX-Subject-Aliases": "user:alice,alice@example.test",
        "X-NHX-Scopes": "models.read",
    }


@pytest.mark.parametrize("token_kind", ["oidc_access_token", "access_key", "workload_subject_token"])
def test_non_workload_access_tokens_ignore_actor_claims(token_kind: ResolvedTokenKind) -> None:
    resolved = ResolvedBearerToken(claims=_claims_with_actor(), token_kind=token_kind)

    principal = resolved.principal

    assert principal.id == "user:alice"
    assert principal.email == "alice@example.test"
    assert principal.groups == ["researchers"]
    assert principal.on_behalf_of is None
    assert principal.on_behalf_of_email is None
    assert principal.on_behalf_of_groups is None
    assert resolved.principal_headers() == {
        "X-NHX-Principal-Id": "user:alice",
        "X-NHX-Principal-Email": "alice@example.test",
        "X-NHX-Principal-Groups": "researchers",
        "X-NHX-Actor-Aliases": "user:alice,alice@example.test",
        "X-NHX-Scopes": "models.read",
    }


@pytest.mark.asyncio
async def test_resolver_skips_access_key_validator_when_access_keys_are_disabled() -> None:
    config = AuthConfig(
        enabled=True,
        access_keys=AccessKeyConfig(enabled=False),
        oidc=OIDCConfig(enabled=False),
    )

    with patch("nhx.common.auth.access_keys.validate_access_key_token") as validate_access_key:
        resolved = await resolve_bearer_token(config, "bearer-token")

    assert resolved is None
    validate_access_key.assert_not_called()


@pytest.mark.asyncio
async def test_resolver_uses_access_key_validator_when_access_keys_are_enabled() -> None:
    config = AuthConfig(
        enabled=True,
        access_keys=AccessKeyConfig(enabled=True),
        oidc=OIDCConfig(enabled=False),
    )
    claims = _claims()

    with patch(
        "nhx.common.auth.access_keys.validate_access_key_token",
        new=AsyncMock(return_value=claims),
    ) as validate_access_key:
        resolved = await resolve_bearer_token(config, "scoped-access-key")

    assert resolved == ResolvedBearerToken(claims=claims, token_kind="access_key")
    assert resolved is not None
    assert resolved.principal.id == "alice@example.com"
    assert resolved.principal_headers() == {
        "X-NHX-Principal-Id": "alice@example.com",
        "X-NHX-Principal-Email": "alice@example.com",
        "X-NHX-Principal-Groups": "team-ml",
        "X-NHX-Actor-Aliases": "alice@example.com",
        "X-NHX-Scopes": "models:read",
    }
    validate_access_key.assert_awaited_once_with(config, "scoped-access-key")


@pytest.mark.asyncio
async def test_resolver_uses_extra_resolvers_before_oidc() -> None:
    config = AuthConfig(
        enabled=True,
        access_keys=AccessKeyConfig(enabled=False),
        oidc=OIDCConfig(enabled=True, issuer="https://sso.example.com", client_id="nemo-helix-cli"),
    )
    workload_claims = _claims("system:serviceaccount:nemo:job")
    extra_resolver = AsyncMock(
        return_value=ResolvedBearerToken(
            claims=workload_claims,
            token_kind="workload_access_token",
        )
    )
    jwt_validator = MagicMock()

    resolved = await resolve_bearer_token(
        config,
        "workload-token",
        jwt_validator=jwt_validator,
        extra_resolvers=[extra_resolver],
    )

    assert resolved == ResolvedBearerToken(claims=workload_claims, token_kind="workload_access_token")
    extra_resolver.assert_awaited_once_with("workload-token")
    jwt_validator.validate_token.assert_not_called()


@pytest.mark.asyncio
async def test_resolver_falls_back_to_oidc_validator() -> None:
    config = AuthConfig(
        enabled=True,
        access_keys=AccessKeyConfig(enabled=True),
        oidc=OIDCConfig(enabled=True, issuer="https://sso.example.com", client_id="nemo-helix-cli"),
    )
    oidc_claims = _claims("bob@example.com")
    jwt_validator = MagicMock()
    jwt_validator.validate_token = AsyncMock(return_value=oidc_claims)

    with patch(
        "nhx.common.auth.access_keys.validate_access_key_token",
        new=AsyncMock(return_value=None),
    ):
        resolved = await resolve_bearer_token(config, "oidc-token", jwt_validator=jwt_validator)

    assert resolved == ResolvedBearerToken(claims=oidc_claims, token_kind="oidc_access_token")
    jwt_validator.validate_token.assert_awaited_once_with("oidc-token")

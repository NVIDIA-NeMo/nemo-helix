# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.common.auth.json_payload import JsonObject
from nhx.common.auth.token_claims import ActorClaims, TokenClaimsExtractor
from nhx.common.config import AuthConfig
from nhx.common.config.base import OIDCConfig


@pytest.fixture
def auth_config() -> AuthConfig:
    return AuthConfig(
        enabled=True,
        policy_decision_point_base_url="http://localhost:8181",
        oidc=OIDCConfig(
            enabled=True,
            issuer="https://sso.example.com",
            client_id="test-client",
            email_claim="mail",
            groups_claim="roles",
            subject_claim="preferred_username",
            scope_prefix="api://nhx/",
        ),
    )


def test_token_claims_extractor_projects_configured_claims(auth_config: AuthConfig) -> None:
    claims: JsonObject = {
        "preferred_username": "alice",
        "mail": "alice@example.com",
        "roles": "admins, developers",
        "scope": "api://nhx/models:read platform:write openid profile.email",
        "act": {
            "sub": "system:serviceaccount:nemo-runs:job-runner",
            "roles": [" system:serviceaccounts ", 42, "nemo-jobs"],
        },
    }

    token_claims = TokenClaimsExtractor(auth_config).extract(claims)

    assert token_claims is not None
    assert token_claims.subject == "alice"
    assert token_claims.email == "alice@example.com"
    assert token_claims.groups == ["admins", "developers"]
    assert token_claims.scopes == ["models:read", "platform:write"]
    assert token_claims.raw_claims is claims
    assert token_claims.actor == ActorClaims(
        subject="system:serviceaccount:nemo-runs:job-runner",
        groups=["system:serviceaccounts", "nemo-jobs"],
    )


def test_token_claims_extractor_uses_cognito_groups_fallback(auth_config: AuthConfig) -> None:
    claims: JsonObject = {
        "preferred_username": "alice",
        "cognito:groups": ["admins", 42, " developers "],
    }

    token_claims = TokenClaimsExtractor(auth_config).extract(claims)

    assert token_claims is not None
    assert token_claims.groups == ["admins", "developers"]


def test_token_claims_extractor_projects_configured_role_map_claim(auth_config: AuthConfig) -> None:
    claims: JsonObject = {
        "preferred_username": "alice",
        "roles": {
            "admins": {"123": "NeMo Helix"},
            " developers ": {},
        },
    }

    token_claims = TokenClaimsExtractor(auth_config).extract(claims)

    assert token_claims is not None
    assert token_claims.groups == ["admins", "developers"]


def test_token_claims_extractor_ignores_unknown_scopes_after_prefix_stripping() -> None:
    config = AuthConfig(
        enabled=True,
        policy_decision_point_base_url="http://localhost:8181",
        oidc=OIDCConfig(
            enabled=True,
            issuer="https://sso.example.com",
            client_id="test-client",
            scope_prefix="api://primary/",
        ),
    )
    claims: JsonObject = {
        "sub": "alice",
        "scope": "openid profile.email api://primary/models:read custom.audit.read",
    }

    token_claims = TokenClaimsExtractor(config).extract(claims)

    assert token_claims is not None
    assert token_claims.scopes == ["models:read"]


def test_token_claims_extractor_keeps_valid_nhx_scopes_even_when_unknown_to_platform(auth_config: AuthConfig) -> None:
    claims: JsonObject = {
        "preferred_username": "alice",
        "scope": "api://nhx/not-a-real-service:read imaginary:write",
    }

    token_claims = TokenClaimsExtractor(auth_config).extract(claims)

    assert token_claims is not None
    assert token_claims.scopes == ["not-a-real-service:read", "imaginary:write"]


def test_token_claims_extractor_returns_none_without_valid_subject(auth_config: AuthConfig) -> None:
    assert TokenClaimsExtractor(auth_config).extract({"preferred_username": ""}) is None

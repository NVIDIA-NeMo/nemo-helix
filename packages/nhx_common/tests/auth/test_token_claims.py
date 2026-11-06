# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.common.auth.json_payload import JsonObject
from nhx.common.auth.token_claims import ActorClaims, TokenClaimsExtractor
from nhx.common.config import AuthConfig
from nhx.common.config.base import OIDCConfig, OIDCPublicClientConfig


@pytest.fixture
def auth_config() -> AuthConfig:
    return AuthConfig(
        enabled=True,
        policy_decision_point_base_url="http://localhost:8181",
        oidc=OIDCConfig(
            enabled=True,
            issuer="https://sso.example.com",
            public_client=OIDCPublicClientConfig(client_id="test-client", scope_prefix="nhx:"),
            email_claim="mail",
            groups_claim="roles",
            subject_claim="preferred_username",
        ),
    )


def test_token_claims_extractor_projects_configured_claims(auth_config: AuthConfig) -> None:
    claims: JsonObject = {
        "preferred_username": "alice",
        "mail": "alice@example.com",
        "email_verified": True,
        "roles": "admins, developers",
        "scope": "nhx:read write",
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
    assert token_claims.scopes == ["read", "write"]
    assert token_claims.raw_claims is claims
    assert token_claims.actor == ActorClaims(
        subject="system:serviceaccount:nemo-runs:job-runner",
        groups=["system:serviceaccounts", "nemo-jobs"],
    )


def test_token_claims_extractor_preserves_external_scope_list(auth_config: AuthConfig) -> None:
    scope_claim = [
        "vendor:tenant:service:Admin",
        "app.default",
        "licenses.read",
        "openid",
        "profile",
        "email",
    ]
    claims: JsonObject = {
        "preferred_username": "alice",
        "scope": scope_claim,
    }

    token_claims = TokenClaimsExtractor(auth_config).extract(claims)

    assert token_claims is not None
    assert token_claims.scopes == scope_claim


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


def test_token_claims_extractor_returns_none_without_valid_subject(auth_config: AuthConfig) -> None:
    assert TokenClaimsExtractor(auth_config).extract({"preferred_username": ""}) is None


@pytest.mark.parametrize(
    "extra_claims",
    [{}, {"email_verified": False}, {"email_verified": "false"}, {"email_verified": None}, {"email_verified": 1}],
)
def test_token_claims_extractor_drops_unverified_email(auth_config: AuthConfig, extra_claims: JsonObject) -> None:
    claims: JsonObject = {"preferred_username": "mallory", "mail": "victim-admin@corp.test", **extra_claims}

    token_claims = TokenClaimsExtractor(auth_config).extract(claims)

    assert token_claims is not None
    assert token_claims.subject == "mallory"
    assert token_claims.email is None


@pytest.mark.parametrize("verified", [True, "true", "True"])
def test_token_claims_extractor_keeps_verified_email(auth_config: AuthConfig, verified: object) -> None:
    claims: JsonObject = {"preferred_username": "alice", "mail": "alice@example.com", "email_verified": verified}

    token_claims = TokenClaimsExtractor(auth_config).extract(claims)

    assert token_claims is not None
    assert token_claims.email == "alice@example.com"


def test_token_claims_extractor_unverified_email_never_becomes_authz_alias(auth_config: AuthConfig) -> None:
    from nhx.common.auth.token_resolver import ResolvedBearerToken

    claims: JsonObject = {"preferred_username": "mallory", "mail": "victim-admin@corp.test", "email_verified": False}
    token_claims = TokenClaimsExtractor(auth_config).extract(claims)
    assert token_claims is not None

    principal = ResolvedBearerToken(claims=token_claims, token_kind="oidc_access_token").principal

    assert principal.email is None
    assert principal.authz_aliases == ["mallory"]


def test_token_claims_extractor_can_opt_out_of_email_verification(auth_config: AuthConfig) -> None:
    auth_config.oidc.require_verified_email = False
    claims: JsonObject = {"preferred_username": "alice", "mail": "alice@example.com"}

    token_claims = TokenClaimsExtractor(auth_config).extract(claims)

    assert token_claims is not None
    assert token_claims.email == "alice@example.com"


def test_token_claims_extractor_rejects_unverified_email_used_as_subject(auth_config: AuthConfig) -> None:
    auth_config.oidc.subject_claim = "mail"

    assert TokenClaimsExtractor(auth_config).extract({"mail": "victim-admin@corp.test"}) is None
    assert TokenClaimsExtractor(auth_config).extract({"mail": "victim-admin@corp.test", "email_verified": True})

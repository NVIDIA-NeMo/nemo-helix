# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nmp.common.auth.json_payload import JsonObject
from nmp.common.auth.token_claims import ActorClaims, TokenClaimsExtractor
from nmp.common.config import AuthConfig
from nmp.common.config.base import OIDCConfig


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
            scope_prefix="nmp:",
        ),
    )


def test_token_claims_extractor_projects_configured_claims(auth_config: AuthConfig) -> None:
    claims: JsonObject = {
        "preferred_username": "alice",
        "mail": "alice@example.com",
        "roles": "admins, developers",
        "scope": "nmp:read write",
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
            "admins": {"123": "NeMo Platform"},
            " developers ": {},
        },
    }

    token_claims = TokenClaimsExtractor(auth_config).extract(claims)

    assert token_claims is not None
    assert token_claims.groups == ["admins", "developers"]


def test_token_claims_extractor_returns_none_without_valid_subject(auth_config: AuthConfig) -> None:
    assert TokenClaimsExtractor(auth_config).extract({"preferred_username": ""}) is None

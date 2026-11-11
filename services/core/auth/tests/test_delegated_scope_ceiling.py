# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delegation must not let an auth-capable key mint a wider inference key."""

import pytest
from nhx.common.auth import AuthClient
from nhx.common.auth.token_claims import ActorClaims, TokenClaims
from nhx.common.auth.token_resolver import ResolvedBearerToken, ResolvedTokenKind
from nhx.common.config import AuthConfig
from nhx.core.auth.api.v2.access_keys.endpoints import _caller_access_key_scope


@pytest.mark.parametrize("kind", ["access_key", "workload_access_token"])
def test_key_issuance_keeps_original_service_ceiling(kind: ResolvedTokenKind) -> None:
    scopes = ["evals:write", "auth:write"]
    claims = TokenClaims(
        subject="creator",
        email=None,
        groups=[],
        scopes=scopes,
        actor=ActorClaims(subject="service:evals", groups=[]) if kind == "workload_access_token" else None,
        raw_claims={"nhx_credential_scopes": scopes} if kind == "workload_access_token" else {},
    )
    resolved = ResolvedBearerToken(claims=claims, token_kind=kind)
    client = AuthClient(principal=resolved.principal, config=AuthConfig(enabled=True), resolved_bearer_token=resolved)
    ceiling = _caller_access_key_scope(client)
    assert ceiling == ["auth", "evals"]
    assert ceiling is not None and "inference" not in ceiling


def test_unrestricted_oidc_workload_has_no_access_key_ceiling() -> None:
    claims = TokenClaims(
        subject="creator",
        email=None,
        groups=[],
        scopes=["openid"],
        actor=ActorClaims(subject="service:evals", groups=[]),
        raw_claims={},
    )
    resolved = ResolvedBearerToken(claims=claims, token_kind="workload_access_token")
    client = AuthClient(principal=resolved.principal, config=AuthConfig(enabled=True), resolved_bearer_token=resolved)
    assert _caller_access_key_scope(client) is None

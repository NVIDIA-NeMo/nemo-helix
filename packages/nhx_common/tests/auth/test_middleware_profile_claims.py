# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.common.auth.middleware import _identity_resolution_from_resolved_token
from nhx.common.auth.token_claims import ActorClaims, TokenClaims
from nhx.common.auth.token_resolver import ResolvedBearerToken
from nhx.common.config import AuthConfig

_PROFILE_CLAIMS: dict[str, object] = {
    "name": "Alice Anderson",
    "picture": "https://cdn.example.com/a.png",
    "address": "secret",
}


def test_direct_token_descriptor_carries_display_claims() -> None:
    resolved = ResolvedBearerToken(
        claims=TokenClaims(
            subject="alice", email="alice@example.com", groups=[], scopes=[], raw_claims=_PROFILE_CLAIMS
        ),
        token_kind="oidc_access_token",
    )

    descriptor = _identity_resolution_from_resolved_token(resolved, AuthConfig())["principal"]

    assert descriptor["name"] == "Alice Anderson"
    assert descriptor["picture"] == "https://cdn.example.com/a.png"
    assert "address" not in descriptor
    assert descriptor["claims_snapshot"] == {}


def test_actor_token_attributes_display_claims_to_the_subject() -> None:
    resolved = ResolvedBearerToken(
        claims=TokenClaims(
            subject="alice",
            email="alice@example.com",
            groups=[],
            scopes=[],
            raw_claims=_PROFILE_CLAIMS,
            actor=ActorClaims(subject="system:serviceaccount:runs:job", groups=[]),
        ),
        token_kind="workload_access_token",
    )

    resolution = _identity_resolution_from_resolved_token(resolved, AuthConfig())

    assert "name" not in resolution["principal"]
    assert resolution["on_behalf_of"]["name"] == "Alice Anderson"
    assert resolution["on_behalf_of"]["picture"] == "https://cdn.example.com/a.png"

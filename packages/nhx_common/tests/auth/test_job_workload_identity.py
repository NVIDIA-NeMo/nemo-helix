# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job capability evidence must come from a verified delegated workload token."""

import pytest
from nhx.common.auth.middleware import _identity_resolution_from_resolved_token
from nhx.common.auth.token_claims import ActorClaims, TokenClaims
from nhx.common.auth.token_resolver import ResolvedBearerToken, ResolvedTokenKind
from nhx.common.config import AuthConfig


@pytest.mark.parametrize(
    "token_kind,delegated,expected",
    [
        ("workload_access_token", True, True),
        ("workload_access_token", False, False),
        ("workload_subject_token", True, False),
        ("oidc_access_token", True, False),
        ("access_key", True, False),
    ],
)
def test_only_verified_delegated_workload_access_tokens_forward_job_evidence(
    token_kind: ResolvedTokenKind, delegated: bool, expected: bool
) -> None:
    workload = {"kind": "job", "workspace": "source", "job_id": "job-1"}
    resolved = ResolvedBearerToken(
        token_kind=token_kind,
        claims=TokenClaims(
            subject="creator@example.com",
            email=None,
            groups=[],
            scopes=[],
            raw_claims={"iss": "https://issuer.example", "nhx_workload": workload},
            actor=ActorClaims(subject="job-runtime", groups=[]) if delegated else None,
        ),
    )
    resolution = _identity_resolution_from_resolved_token(resolved, AuthConfig())
    snapshot = resolution["principal"]["claims_snapshot"]
    assert ("nhx_workload" in snapshot) is expected
    if expected:
        assert snapshot["nhx_workload"] == workload
        assert resolution["principal"]["subject"] == "job-runtime"
        assert resolution["principal"]["subject_claim"] == "act.sub"
        assert resolution["on_behalf_of"]["subject"] == "creator@example.com"

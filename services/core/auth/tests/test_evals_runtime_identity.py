# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verified Auth credentials must bind Evals runtime requests to their Job."""

import pytest


@pytest.mark.parametrize(
    "case",
    [
        "wrong_workspace",
        "wrong_job",
        "missing_binding",
        "malformed_binding",
        "oidc_claim",
        "access_key_claim",
        "undelegated_workload",
    ],
)
async def test_runtime_rejects_unbound_identity_before_reading_any_job(case: str) -> None:
    import httpx
    from fastapi import HTTPException
    from nemo_evals.api.v2.job_runtime import _job
    from nemo_helix_plugin.client.client import AsyncNemoClient
    from nhx.common.auth import AuthClient
    from nhx.common.auth.dependencies import auth_client_context
    from nhx.common.auth.json_payload import JsonObject
    from nhx.common.auth.token_claims import ActorClaims, TokenClaims
    from nhx.common.auth.token_resolver import ResolvedBearerToken, ResolvedTokenKind
    from nhx.common.config import AuthConfig

    binding = {"workspace": "team", "job": "job-one"}
    if case == "wrong_workspace":
        binding["workspace"] = "another-team"
    if case == "wrong_job":
        binding["job"] = "another-job"
    raw: JsonObject = {"nhx_workload": binding}
    if case == "missing_binding":
        raw = {}
    if case == "malformed_binding":
        raw = {"nhx_workload": []}
    kind: ResolvedTokenKind = "workload_access_token"
    if case == "oidc_claim":
        kind = "oidc_access_token"
    if case == "access_key_claim":
        kind = "access_key"
    claims = TokenClaims(
        subject="creator",
        email=None,
        groups=[],
        scopes=["inference:write"],
        actor=None
        if case == "undelegated_workload"
        else ActorClaims(subject="system:serviceaccount:qa:jobs", groups=[]),
        raw_claims=raw,
    )
    resolved = ResolvedBearerToken(claims=claims, token_kind=kind)
    auth = AuthClient(principal=resolved.principal, config=AuthConfig(enabled=True), resolved_bearer_token=resolved)

    def reject_network(request: httpx.Request) -> httpx.Response:
        pytest.fail("An unbound request must be rejected before looking up a Job")

    marker = auth_client_context.set(auth)
    try:
        async with AsyncNemoClient(
            base_url="https://platform.test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(reject_network)),
            owns_http_client=True,
        ) as client:
            with pytest.raises(HTTPException) as error:
                await _job("team", "job-one", auth, client, {"nemo-evals.agent-evaluate"})
            assert error.value.status_code == 403
    finally:
        auth_client_context.reset(marker)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The producing Job's identity survives the runtime client's wire format."""

import pytest
from nemo_evals.entities import AgentEvalResultEntity, EvaluateResultEntity
from nemo_evals.jobs.runtime_client import save_result
from nhx_evals_sdk.values.results import AggregatedMetricResult
from pydantic import TypeAdapter


@pytest.mark.parametrize("case", ["bound", "another_job", "another_workspace", "unbound", "ordinary", "other_service"])
def test_trial_revocation_requires_issuing_job_binding(case: str) -> None:
    from nemo_evals.api.v2.job_runtime import _trial_key_belongs_to_job

    claims = {"kind": "trial", "intake_workspace": "team", "runtime_job": "job-one"}
    if case == "another_job":
        claims["runtime_job"] = "job-two"
    elif case == "another_workspace":
        claims["intake_workspace"] = "another-team"
    elif case == "unbound":
        del claims["runtime_job"]
    elif case == "ordinary":
        claims["kind"] = "ordinary"
    attributes = {"another-service" if case == "other_service" else "evals": claims}
    assert _trial_key_belongs_to_job(attributes, workspace="team", job="job-one") is (case == "bound")


def test_production_runtime_routes_derive_delegated_contracts() -> None:
    from nemo_evals.service import EvaluatorPluginService
    from nemo_helix_plugin.authz_discovery import _derive_service_contribution

    contribution, errors, _ = _derive_service_contribution(EvaluatorPluginService())
    assert not errors
    endpoints = contribution.to_dict()["endpoints"]
    runtime = {path: methods for path, methods in endpoints.items() if "/job-runtime/" in path}
    assert len(runtime) == 4
    for methods in runtime.values():
        for contract in methods.values():
            assert contract["delegated"] == {"actor_perms": [], "subject_perms": []}
            assert "service_principal" in contract["callers"]
            assert contract["scopes"]


@pytest.mark.parametrize("kind", ["agent", "row"])
def test_runtime_result_preserves_producing_job_on_the_wire(kind: str) -> None:
    common = dict(
        name="job-one",
        workspace="team",
        job_id="job-one",
        target_kind=None,
        target_name=None,
        target_url=None,
        scores=AggregatedMetricResult(scores=[]),
        bundle_ref="fileset://team/job-one",
    )
    entity = (
        AgentEvalResultEntity.model_validate(common)
        if kind == "agent"
        else EvaluateResultEntity.model_validate(common | {"dataset_ref": None, "metric_types": []})
    )
    request = save_result(workspace="team", job="job-one", body=entity)
    assert isinstance(request.content, bytes)
    received = TypeAdapter(AgentEvalResultEntity | EvaluateResultEntity).validate_json(request.content)
    assert type(received) is type(entity)
    assert (received.workspace, received.name, received.job_id) == ("team", "job-one", "job-one")


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

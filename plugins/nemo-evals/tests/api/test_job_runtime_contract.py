# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The producing Job's identity survives the runtime client's wire format."""

import httpx
import pytest
from nemo_evals.entities import AgentEvalResultEntity, EvaluateResultEntity
from nemo_evals.jobs.runtime_client import save_result
from nhx_evals_sdk.values.results import AggregatedMetricResult
from pydantic import JsonValue, TypeAdapter


@pytest.mark.parametrize(
    ("status", "issuer_detail", "expose_limit"),
    [
        (400, "expires_in_seconds must be less than or equal to auth.access_keys.max_expires_in_seconds (3600)", True),
        (403, "private permission diagnostic", False),
        (400, "arbitrary issuer response", False),
        (
            400,
            "expires_in_seconds must be less than or equal to auth.access_keys.max_expires_in_seconds (3600) private",
            False,
        ),
        (500, "expires_in_seconds must be less than or equal to auth.access_keys.max_expires_in_seconds (3600)", False),
    ],
)
def test_trial_issuance_preserves_only_public_lifetime_limit(
    status: int, issuer_detail: str, expose_limit: bool
) -> None:
    from nemo_evals.api.v2.job_runtime import _trial_issuance_error
    from nemo_helix_plugin.client.errors import NemoHTTPError

    error = NemoHTTPError(httpx.Response(status, json={"detail": issuer_detail}))
    response = _trial_issuance_error(error)
    assert response.status_code == status
    expected = "Trial credential issuance was refused" + (": " + issuer_detail if expose_limit else "")
    assert response.detail == expected


@pytest.mark.parametrize("case", ["bound", "another_job", "another_workspace", "unbound", "ordinary", "other_service"])
def test_trial_revocation_requires_issuing_job_binding(case: str) -> None:
    from nemo_evals.api.v2.job_runtime import _trial_key_belongs_to_job

    claims: dict[str, JsonValue] = {"kind": "trial", "intake_workspace": "team", "runtime_job": "job-one"}
    if case == "another_job":
        claims["runtime_job"] = "job-two"
    elif case == "another_workspace":
        claims["intake_workspace"] = "another-team"
    elif case == "unbound":
        del claims["runtime_job"]
    elif case == "ordinary":
        claims["kind"] = "ordinary"
    attributes: dict[str, dict[str, JsonValue]] = {"another-service" if case == "other_service" else "evals": claims}
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

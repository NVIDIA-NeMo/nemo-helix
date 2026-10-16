# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path

import pytest
from nemo_helix_plugin.evals import endpoints
from nemo_helix_plugin.evals.client import EvaluatorClient
from nemo_helix_plugin.evals.types import SubmitEvaluateJobRequest
from nhx.testing import grant_workspace_role
from nhx.testing.e2e import wait_for_platform_job

from tests.auth_idp.common import require_capability

pytestmark = [
    pytest.mark.auth_idp,
    pytest.mark.auth_idp_runtime,
    pytest.mark.e2e,
]

_OFFLINE_EXACT_MATCH_SPEC = (
    Path(__file__).parents[3] / "skills" / "nemo-evals-plugin" / "assets" / "specs" / "exact_match_metric.json"
)


def test_evaluation_job_saves_its_result_record(
    auth_idp_case,
    auth_idp_runtime,
    auth_idp_workspace,
):
    """An evals job acts as evals for its submitter, so it may write the service-managed result record."""
    require_capability(auth_idp_case, "workspace_rbac")
    require_capability(auth_idp_case, "managed_workload_job_obo")
    require_capability(auth_idp_case, "evals_job")

    e2e_setup_client = auth_idp_runtime.e2e_setup_client()
    for principal in auth_idp_runtime.workload_role_principals():
        grant_workspace_role(
            e2e_setup_client,
            workspace=auth_idp_workspace,
            principal=principal,
            roles=["Viewer", "Editor", "JobRunner"],
        )

    submitter_client = auth_idp_runtime.workload_provider_client()
    spec = json.loads(_OFFLINE_EXACT_MATCH_SPEC.read_text())
    submit = endpoints.submit_evaluate_job(workspace=auth_idp_workspace, body=SubmitEvaluateJobRequest(spec=spec))
    # The typed request has no ``profile``; the route takes one, and this runtime's only job profile is "workload".
    submit_on_workload_profile = replace(submit, content=json.dumps({"spec": spec, "profile": "workload"}).encode())
    job = submitter_client.send(submit_on_workload_profile).data()
    evals = EvaluatorClient.from_client(submitter_client)

    completed_job = wait_for_platform_job(e2e_setup_client, job.name, auth_idp_workspace, timeout=600)
    assert completed_job.status == "completed"

    results = evals.list_eval_results(workspace=auth_idp_workspace)
    assert [result.model_dump()["job_id"] for result in results] == [job.name]

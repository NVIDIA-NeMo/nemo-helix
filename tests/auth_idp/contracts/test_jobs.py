# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nhx.testing import grant_workspace_role
from nhx.testing.e2e import wait_for_job_logs, wait_for_platform_job

from tests.auth_idp.common import managed_workload_workspace_get_command, nhx_api_image, require_capability

pytestmark = [
    pytest.mark.auth_idp,
    pytest.mark.auth_idp_runtime,
    pytest.mark.e2e,
]


def test_provider_workload_job_runs_via_workload_profile(
    auth_idp_case,
    auth_idp_runtime,
    auth_idp_workspace,
):
    require_capability(auth_idp_case, "workspace_rbac")
    require_capability(auth_idp_case, "workload_job")
    require_capability(auth_idp_case, "managed_workload_job_obo")

    e2e_setup_sdk = auth_idp_runtime.e2e_setup_sdk()
    for principal in auth_idp_runtime.workload_role_principals():
        grant_workspace_role(
            e2e_setup_sdk,
            workspace=auth_idp_workspace,
            principal=principal,
            roles=["Viewer", "Editor", "JobRunner"],
        )

    job_submitter_sdk = auth_idp_runtime.workload_provider_sdk()
    job = (
        client_from_platform(job_submitter_sdk, JobsClient)
        .create_job(
            workspace=auth_idp_workspace,
            body=CreateHelixJobRequest(
                source=f"{auth_idp_case.id}-workload-job",
                spec={"test": "workload-job"},
                platform_spec={
                    "steps": [
                        {
                            "name": "workload-workspace-get",
                            "executor": {
                                "provider": "cpu",
                                "profile": "workload",
                                "container": {
                                    "image": nhx_api_image(),
                                    "entrypoint": ["sh", "-c"],
                                    "command": [
                                        managed_workload_workspace_get_command(),
                                    ],
                                },
                            },
                            "config": {
                                "workspace": auth_idp_workspace,
                            },
                        }
                    ]
                },
            ),
        )
        .data()
    )

    completed_job = wait_for_platform_job(e2e_setup_sdk, job.name, auth_idp_workspace, timeout=240)
    assert completed_job.status == "completed"

    step_logs = wait_for_job_logs(e2e_setup_sdk, job.name, auth_idp_workspace, min_log_count=1, timeout=240)
    assert step_logs.data
    assert all(log.job == job.name for log in step_logs.data)
    assert all(log.job_step == "workload-workspace-get" for log in step_logs.data)
    assert all(log.job_task for log in step_logs.data)
    assert all(log.message.strip() for log in step_logs.data)
    assert any(
        "Workload auth env: NHX_PRINCIPAL=absent NHX_WORKLOAD_IDENTITY_TOKEN_FILE=present" in log.message
        for log in step_logs.data
    )
    assert any(f"Successfully retrieved workspace: {auth_idp_workspace}" in log.message for log in step_logs.data)

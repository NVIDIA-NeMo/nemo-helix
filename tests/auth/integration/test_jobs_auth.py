# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for jobs with auth enabled.

These tests run against the auth-enabled subprocess services pool. The job steps
use the local cpu/default -> subprocess translation, so the suite does not
require a container runtime or Kubernetes cluster.
"""

import logging
from contextlib import ExitStack
from uuid import uuid4

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_ext.auth.helpers import generate_unsigned_jwt
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    EnvironmentVariable,
    HelixJobSpec,
    HelixJobStep,
)
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nhx.common.entities import ALL_WORKSPACES
from nhx.core.jobs.controllers.diagnostics import collect_job_diagnostics
from nhx.testing import TEST_ADMIN_EMAIL, grant_workspace_role, short_unique_name, unique_email
from nhx.testing.e2e import wait_for_platform_job

from tests.auth.integration.jobs_auth_helpers import job_exists_in_pages, managed_admin_workspace

JOB_SOURCE = "integration-auth-test"
logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.subprocess_only,
    pytest.mark.e2e_config("e2e/configs/local-subprocess.yaml", {"auth": {"enabled": True}}),
    # services_pool_sdk is module-scoped: keep these tests on one xdist worker so they
    # share the already-booted service pool instead of each worker re-provisioning it.
    pytest.mark.xdist_group("jobs_auth"),
]


def _as_bearer_user(
    sdk: NeMoHelix,
    email: str,
    *,
    principal_id: str | None = None,
    groups: list[str] | None = None,
) -> NeMoHelix:
    token = generate_unsigned_jwt(
        principal_id=principal_id or email,
        email=email,
        groups=groups,
    )
    return sdk.with_options(set_default_headers={"Authorization": f"Bearer {token}"})


def _oidc_subject() -> str:
    """Return an IdP-style subject id distinct from the user's email."""
    return str(uuid4())


def _log_auth_job_diagnostics(
    sdk: NeMoHelix,
    *,
    workspace: str,
    job_name: str,
    step_name: str,
    context: str,
) -> None:
    logger.error(
        "Auth job diagnostics",
        extra={
            "diagnostic_context": context,
            "workspace": workspace,
            "job_name": job_name,
            "step_name": step_name,
            "job_diagnostics": collect_job_diagnostics(
                sdk,
                workspace=workspace,
                job_name=job_name,
                step_name=step_name,
                context=context,
            ),
        },
    )


def test_job_principal_propagation(services_pool_sdk: NeMoHelix):
    admin_sdk = _as_bearer_user(services_pool_sdk, TEST_ADMIN_EMAIL, groups=["admin"])
    user_email = unique_email("job-creator")
    workspace_name = short_unique_name("job-auth-test")

    with managed_admin_workspace(admin_sdk, workspace_name):
        grant_workspace_role(admin_sdk, workspace=workspace_name, principal=user_email, roles=["Editor"])

        user_sdk = _as_bearer_user(services_pool_sdk, user_email, principal_id=_oidc_subject())
        job = (
            client_from_platform(user_sdk, JobsClient)
            .create_job(
                workspace=workspace_name,
                body=CreateHelixJobRequest(
                    source=JOB_SOURCE,
                    spec={"test": "auth-propagation"},
                    platform_spec=HelixJobSpec(
                        steps=[
                            HelixJobStep(
                                name="auth-test-step",
                                executor=CPUExecutionProviderSpec(
                                    provider="cpu",
                                    container=ContainerSpec(
                                        entrypoint=["nemo-helix"],
                                        command=["run", "task", "--task", "nhx.hello_world.tasks.hello_world"],
                                    ),
                                ),
                                environment=[EnvironmentVariable(name="BUSY_LOOP_DURATION_SECONDS", value="0")],
                                config={"message": "auth propagation test"},
                            )
                        ]
                    ),
                ),
            )
            .data()
        )

        completed_job = wait_for_platform_job(user_sdk, job.name, workspace_name)
        assert completed_job.status == "completed"

        fileset_name = f"hello-world-{job.name}"
        files = client_from_platform(user_sdk, FilesClient)
        fileset = files.get_fileset(workspace=workspace_name, name=fileset_name).data()
        assert fileset is not None

        file_content = user_sdk.files.download_content(
            remote_path="message.txt",
            fileset=fileset_name,
            workspace=workspace_name,
        )
        assert file_content == b"auth propagation test"


def test_job_cannot_access_unauthorized_workspace(services_pool_sdk: NeMoHelix):
    admin_sdk = _as_bearer_user(services_pool_sdk, TEST_ADMIN_EMAIL, groups=["admin"])
    owner_email = unique_email("owner")
    other_email = unique_email("other")

    restricted_workspace = short_unique_name("restricted")
    runner_workspace = short_unique_name("runner")

    with ExitStack() as stack:
        stack.enter_context(managed_admin_workspace(admin_sdk, restricted_workspace))
        stack.enter_context(managed_admin_workspace(admin_sdk, runner_workspace))
        grant_workspace_role(admin_sdk, workspace=restricted_workspace, principal=owner_email, roles=["Editor"])
        grant_workspace_role(admin_sdk, workspace=runner_workspace, principal=other_email, roles=["Editor"])

        owner_sdk = _as_bearer_user(services_pool_sdk, owner_email, principal_id=_oidc_subject())
        other_sdk = _as_bearer_user(services_pool_sdk, other_email, principal_id=_oidc_subject())

        fileset_name = "private-data"
        files = client_from_platform(owner_sdk, FilesClient)
        files.create_fileset(workspace=restricted_workspace, body=CreateFilesetRequest(name=fileset_name))

        job = (
            client_from_platform(other_sdk, JobsClient)
            .create_job(
                workspace=runner_workspace,
                body=CreateHelixJobRequest(
                    source=JOB_SOURCE,
                    spec={"test": "auth-denial"},
                    platform_spec=HelixJobSpec(
                        steps=[
                            HelixJobStep(
                                name="access-test-step",
                                executor=CPUExecutionProviderSpec(
                                    provider="cpu",
                                    container=ContainerSpec(
                                        entrypoint=["nemo-helix"],
                                        command=["run", "task", "--task", "nhx.hello_world.tasks.access_fileset"],
                                    ),
                                ),
                                config={
                                    "workspace": restricted_workspace,
                                    "fileset": fileset_name,
                                },
                            )
                        ]
                    ),
                ),
            )
            .data()
        )

        completed_job = wait_for_platform_job(other_sdk, job.name, runner_workspace)
        if completed_job.status != "error":
            _log_auth_job_diagnostics(
                other_sdk,
                workspace=runner_workspace,
                job_name=job.name,
                step_name="access-test-step",
                context="expected job to fail with unauthorized workspace access",
            )
        assert completed_job.status == "error"

        tasks_response = (
            client_from_platform(other_sdk, JobsClient)
            .list_job_step_tasks(name="access-test-step", job=job.name, workspace=runner_workspace)
            .data()
        )
        if not tasks_response.data:
            _log_auth_job_diagnostics(
                other_sdk,
                workspace=runner_workspace,
                job_name=job.name,
                step_name="access-test-step",
                context="expected task list to include failed access task",
            )
        assert tasks_response.data
        task = tasks_response.data[0]
        if not task.error_stack or "403" not in task.error_stack or "Forbidden" not in task.error_stack:
            _log_auth_job_diagnostics(
                other_sdk,
                workspace=runner_workspace,
                job_name=job.name,
                step_name="access-test-step",
                context="expected task error stack to include 403 forbidden details",
            )
        assert task.error_stack
        assert "403" in task.error_stack and "Forbidden" in task.error_stack


def test_job_admin_can_list_jobs_in_all_workspaces(services_pool_sdk: NeMoHelix):
    admin_sdk = _as_bearer_user(services_pool_sdk, TEST_ADMIN_EMAIL, groups=["admin"])
    user_email = unique_email("member")
    workspace_name = short_unique_name("admin-list-jobs")

    with managed_admin_workspace(admin_sdk, workspace_name):
        grant_workspace_role(admin_sdk, workspace=workspace_name, principal=user_email, roles=["Editor"])

        user_sdk = _as_bearer_user(services_pool_sdk, user_email, principal_id=_oidc_subject())
        job = (
            client_from_platform(user_sdk, JobsClient)
            .create_job(
                workspace=workspace_name,
                body=CreateHelixJobRequest(
                    source=JOB_SOURCE,
                    spec={"test": "admin-list"},
                    platform_spec=HelixJobSpec(
                        steps=[
                            HelixJobStep(
                                name="admin-list-step",
                                executor=CPUExecutionProviderSpec(
                                    provider="cpu",
                                    container=ContainerSpec(
                                        command=["echo", "admin list jobs"],
                                    ),
                                ),
                            )
                        ]
                    ),
                ),
            )
            .data()
        )

        completed_job = wait_for_platform_job(user_sdk, job.name, workspace_name)
        assert completed_job.status == "completed"

        jobs = client_from_platform(admin_sdk, JobsClient).list_jobs(workspace=ALL_WORKSPACES)
        assert job_exists_in_pages(jobs.items(), job.name)

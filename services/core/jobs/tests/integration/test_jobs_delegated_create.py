# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job creates made for a user, directly or by a service on the user's behalf, with auth enabled.

``create_test_client`` builds the app the way the platform runner does, merging every service into
one app. The fixture removes the seeded wildcard Viewer binding on the ``system`` workspace, as on a
``nemo services run`` platform that does not run the platform seed, so users cannot read ``system``.
Jobs must still check for pending storage reclaims there as ``service:jobs``.
"""

from typing import Generator

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceMemberRequest, CreateWorkspaceRequest
from nhx.common.auth.models import Principal
from nhx.common.entities import SYSTEM_WORKSPACE
from nhx.core.files.service import FilesService
from nhx.core.jobs.service import JobsService
from nhx.testing import TEST_ADMIN_EMAIL, as_user, create_test_client, short_unique_name, unique_email

# Module-scoped auth-enabled client: keep this file's tests on one xdist worker.
pytestmark = [pytest.mark.integration, pytest.mark.xdist_group("jobs_delegated_create")]

SERVICE_PRINCIPAL = "service:integration-test"


@pytest.fixture(scope="module")
def client() -> Generator[NemoClient, None, None]:
    with create_test_client(JobsService, FilesService, auth_enabled=True, client_type=NemoClient) as client:
        WorkspacesClient.from_client(as_user(client, TEST_ADMIN_EMAIL)).delete_workspace_member(
            workspace=SYSTEM_WORKSPACE, principal_id="*", query_params={"wait_role_propagation": True}
        )
        yield client


@pytest.fixture
def group_workspace(client: NemoClient) -> tuple[str, str]:
    """A new workspace where only ``group`` is bound, as Editor."""
    workspace = short_unique_name("jobs-obo")
    group = f"group-{short_unique_name('jobs')}"
    admin = WorkspacesClient.from_client(as_user(client, TEST_ADMIN_EMAIL))
    admin.create_workspace(body=CreateWorkspaceRequest(name=workspace))
    admin.create_workspace_member(
        workspace=workspace,
        body=CreateWorkspaceMemberRequest(principal=group, roles=["Editor"]),
        query_params={"wait_role_propagation": True},
    )
    return workspace, group


def _on_behalf_of(client: NemoClient, user: Principal) -> NemoClient:
    """``SERVICE_PRINCIPAL`` acting for ``user``, with the headers ``get_async_nemo_client(on_behalf_of=...)`` sends."""
    return client.with_headers(
        Principal(
            id=SERVICE_PRINCIPAL,
            authz_aliases=[SERVICE_PRINCIPAL],
            on_behalf_of=user.id,
            on_behalf_of_email=user.email,
            on_behalf_of_groups=user.groups,
            on_behalf_of_authz_aliases=user.authz_aliases,
        ).get_headers()
    )


def _job(name: str) -> CreateHelixJobRequest:
    return CreateHelixJobRequest.model_validate(
        {
            "name": name,
            "source": "delegated-create-test",
            "spec": {},
            "platform_spec": {
                "steps": [
                    {
                        "name": "run",
                        "executor": {
                            "provider": "cpu",
                            "profile": "default",
                            "container": {"image": "busybox:latest", "command": ["true"]},
                        },
                    }
                ]
            },
        }
    )


def test_user_creates_job(client: NemoClient, group_workspace: tuple[str, str]) -> None:
    workspace, group = group_workspace
    user = as_user(client, unique_email("creator"), groups=[group])
    name = short_unique_name("direct")

    job = JobsClient.from_client(user).create_job(workspace=workspace, body=_job(name)).data()

    assert job.name == name


def test_service_creates_job_for_user(client: NemoClient, group_workspace: tuple[str, str]) -> None:
    workspace, group = group_workspace
    email = unique_email("creator")
    user = Principal(id=email, email=email, groups=[group], authz_aliases=[email])
    name = short_unique_name("delegated")

    job = JobsClient.from_client(_on_behalf_of(client, user)).create_job(workspace=workspace, body=_job(name)).data()

    assert job.name == name
    # Only service principals see a step's auth_context, which the task runs as.
    service = JobsClient.from_client(as_user(client, SERVICE_PRINCIPAL))
    steps = list(service.list_steps(name=name, workspace=workspace).items())
    assert len(steps) == 1
    auth_context = steps[0].auth_context
    assert auth_context is not None
    assert auth_context.principal_id == SERVICE_PRINCIPAL
    assert auth_context.principal_on_behalf_of == email
    assert auth_context.principal_on_behalf_of_groups == [group]


def test_service_create_for_user_without_access_is_forbidden(
    client: NemoClient, group_workspace: tuple[str, str]
) -> None:
    """The user's group is not delegated, so the user has no role in the workspace."""
    workspace, _ = group_workspace
    email = unique_email("outsider")
    user = Principal(id=email, email=email, authz_aliases=[email])

    with pytest.raises(PermissionDeniedError, match="not allowed to create jobs"):
        JobsClient.from_client(_on_behalf_of(client, user)).create_job(
            workspace=workspace, body=_job(short_unique_name("forbidden"))
        )

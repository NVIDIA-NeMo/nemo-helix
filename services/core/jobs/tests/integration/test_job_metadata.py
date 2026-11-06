# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata HTTP authorization through the real middleware and embedded PDP."""

from collections.abc import Generator

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_helix_plugin.entities import SyncEntityClient
from nemo_helix_plugin.entities.client import EntitiesClient
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.selectors import JobMetadataReferenceKind
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceMemberRequest, CreateWorkspaceRequest
from nhx.common.auth import AuthContext
from nhx.core.files.service import FilesService
from nhx.core.jobs.app.test_helpers import TestConstants
from nhx.core.jobs.entities import HelixJob
from nhx.core.jobs.service import JobsService
from nhx.testing import TEST_ADMIN_EMAIL, as_user, create_test_client, short_unique_name

pytestmark = [pytest.mark.integration, pytest.mark.xdist_group("job_metadata")]


@pytest.fixture(scope="module")
def metadata_platform() -> Generator[tuple[NemoClient, str, HelixJob]]:
    with create_test_client(JobsService, FilesService, auth_enabled=True, client_type=NemoClient) as client:
        workspace = short_unique_name("metadata")
        admin = WorkspacesClient.from_client(as_user(client, TEST_ADMIN_EMAIL))
        admin.create_workspace(body=CreateWorkspaceRequest(name=workspace))
        for principal, role in (
            ("metadata-reader@example.com", "Viewer"),
            ("metadata-writer@example.com", "JobRunner"),
        ):
            admin.create_workspace_member(
                workspace=workspace,
                body=CreateWorkspaceMemberRequest(principal=principal, roles=[role]),
                query_params={"wait_role_propagation": True},
            )
        service = as_user(client, "service:integration-test")
        store = SyncEntityClient(EntitiesClient.from_client(service))
        job = HelixJob(
            name="not-started",
            workspace=workspace,
            fileset="test-fileset",
            source="metadata-test",
            spec={"kind": "test"},
            platform_spec=TestConstants.PLATFORM_SPEC,
        )
        job.with_auth_context(AuthContext(principal_id="creator@example.com"))
        job = store.create(job)
        yield client, workspace, job


@pytest.mark.parametrize("principal", ["metadata-reader@example.com", "service:integration-test"])
@pytest.mark.parametrize("scope", ["jobs:read", "platform:read"])
def test_metadata_accepts_existing_user_and_service_read_access(metadata_platform, principal: str, scope: str):
    client, workspace, job = metadata_platform
    jobs = JobsClient.from_client(as_user(client, principal).with_headers({"X-NHX-Scopes": scope}))
    metadata = jobs.get_job_metadata(
        job=job.id, workspace=workspace, job_reference_kind=JobMetadataReferenceKind.ID
    ).data()
    assert metadata.id == job.id and metadata.parent_id is None
    assert metadata.spec == job.spec
    assert "auth_context" not in metadata.model_dump() and "_auth_context" not in metadata.model_dump()
    assert "platform_spec" not in metadata.model_dump()
    # Attempt-independent lookup leaves the existing get lifecycle unchanged.
    with pytest.raises(NotFoundError):
        jobs.get_job_by_id(job_id=job.id, workspace=workspace)


@pytest.mark.parametrize(
    "principal,scope",
    [
        ("metadata-reader@example.com", "jobs:write"),
        ("metadata-writer@example.com", "jobs:read"),
        ("outsider@example.com", "jobs:read"),
    ],
)
def test_metadata_denies_missing_scope_permission_or_membership(metadata_platform, principal: str, scope: str):
    client, workspace, job = metadata_platform
    jobs = JobsClient.from_client(as_user(client, principal).with_headers({"X-NHX-Scopes": scope}))
    with pytest.raises(PermissionDeniedError):
        jobs.get_job_metadata(job=job.id, workspace=workspace, job_reference_kind=JobMetadataReferenceKind.ID)


def test_metadata_rejects_foreign_workspace_id_even_with_service_access(metadata_platform):
    client, _, job = metadata_platform
    jobs = JobsClient.from_client(as_user(client, "service:integration-test"))
    with pytest.raises(NotFoundError):
        jobs.get_job_metadata(job=job.id, workspace="default", job_reference_kind=JobMetadataReferenceKind.ID)

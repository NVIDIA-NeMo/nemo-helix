# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for job auth context propagation.

These tests verify GitLab issue #3390 Gap 2: jobs should run with the creating
user's auth context, not as anonymous or service principal.

Auth context is only visible to service principals (principal ID starting with "service:")
and when auth is disabled. Regular users see auth_context stripped from responses.

Note: These tests verify job creation with authenticated users works correctly.
Full verification of auth propagation to container environment variables is
done in unit tests (test_docker_backend.py, test_kubernetes_common.py).
"""

from typing import Generator

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nhx.core.files.service import FilesService
from nhx.core.jobs.service import JobsService
from nhx.testing import as_user, create_test_client, short_unique_name, unique_email

# client is module-scoped (expensive to boot, auth_enabled=True): keep this file's tests on
# one xdist worker so they share it instead of each worker re-provisioning it from scratch.
pytestmark = pytest.mark.xdist_group("jobs_auth_propagation")


@pytest.fixture(scope="module")
def client() -> Generator[NemoClient, None, None]:
    """Typed client with JobsService and FilesService (auth enabled).

    Jobs service requires FilesService for fileset creation (job storage).
    """
    with create_test_client(
        JobsService,
        FilesService,
        auth_enabled=True,
        client_type=NemoClient,
    ) as client:
        yield client


def _as_service_principal(client: NemoClient, service_name: str = "jobs-controller") -> NemoClient:
    """Create a typed client authenticated as a service principal."""
    return as_user(client, f"service:{service_name}")


class TestJobCreationWithAuth:
    def test_auth_context_stripped_for_regular_user(self, client: NemoClient):
        """Regular users should not see auth_context in step responses."""
        creator_email = unique_email("creator")
        workspace = "default"
        job_name = short_unique_name("auth-strip-test")

        creator_client = as_user(client, creator_email, groups=["team-alpha"])

        jobs = JobsClient.from_client(creator_client)
        jobs.create_job(
            workspace=workspace,
            body=CreateHelixJobRequest(
                name=job_name,
                source="auth-propagation-test",
                spec={},
                platform_spec={
                    "steps": [
                        {
                            "name": "test-step",
                            "executor": {
                                "provider": "cpu",
                                "profile": "default",
                                "container": {
                                    "image": "busybox:latest",
                                    "entrypoint": ["entrypoint"],
                                    "command": ["command"],
                                },
                            },
                        },
                    ]
                },
            ),
        )

        steps = list(jobs.list_steps(name=job_name, workspace=workspace).items())
        assert len(steps) == 1
        assert steps[0].auth_context is None, "Regular user should not see auth_context"

    def test_auth_context_visible_to_service_principal(self, client: NemoClient):
        """Service principals should see auth_context with the creator's identity."""
        creator_email = unique_email("creator")
        creator_groups = ["team-alpha", "ml-engineers"]
        workspace = "default"
        job_name = short_unique_name("auth-ctx-test")

        creator_client = as_user(client, creator_email, groups=creator_groups)

        jobs = JobsClient.from_client(creator_client)
        jobs.create_job(
            workspace=workspace,
            body=CreateHelixJobRequest(
                name=job_name,
                source="auth-propagation-test",
                spec={},
                platform_spec={
                    "steps": [
                        {
                            "name": "test-step",
                            "executor": {
                                "provider": "cpu",
                                "profile": "default",
                                "container": {
                                    "image": "busybox:latest",
                                    "entrypoint": ["entrypoint"],
                                    "command": ["command"],
                                },
                            },
                        },
                    ]
                },
            ),
        )

        service_client = _as_service_principal(client)
        steps = list(JobsClient.from_client(service_client).list_steps(name=job_name, workspace=workspace).items())

        assert len(steps) == 1
        step = steps[0]
        assert step.auth_context is not None, "Service principal should see auth_context"
        assert step.auth_context.principal_id == creator_email
        assert step.auth_context.principal_email == creator_email
        assert step.auth_context.principal_groups == creator_groups

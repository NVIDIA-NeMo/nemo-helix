# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.common.auth import Principal
from nhx.common.service.dependencies import get_entity_client
from nhx.core.files.service import FilesService
from nhx.core.jobs.service import JobsService
from nhx.testing import TEST_ADMIN_EMAIL, create_test_client, unique_email
from nhx.testing.client import ClientContext


def test_job_groups_routes_use_workspace_permissions() -> None:
    workspace = "job-group-registration"
    path = f"/apis/jobs/v2/workspaces/{workspace}/job-groups"
    admin = Principal(id=TEST_ADMIN_EMAIL, email=TEST_ADMIN_EMAIL).get_headers()
    delegated_admin = Principal(
        id="service:evals", on_behalf_of=TEST_ADMIN_EMAIL, on_behalf_of_email=TEST_ADMIN_EMAIL
    ).get_headers()
    outsider = unique_email("job-group-outsider")
    denied = Principal(id=outsider, email=outsider).get_headers()
    submission = {
        "name": "group-registration",
        "source": "registration",
        "template": {
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
        "members": [{}],
    }
    jobs_service = JobsService()
    with create_test_client(
        lambda: jobs_service,
        FilesService,
        auth_enabled=True,
        client_type=ClientContext,
        access_log=True,
        dependency_overrides={get_entity_client: jobs_service.dependency_provider.get_entity_client},
    ) as clients:
        WorkspacesClient.from_client(clients.client.with_headers(admin)).create_workspace(
            body=CreateWorkspaceRequest(name=workspace)
        )
        created = clients.test_client.post(path, headers=admin, json=submission)
        assert created.status_code == 201, created.text
        group_id = created.json()["id"]
        listed = clients.test_client.get(path, headers=admin)
        assert listed.status_code == 200, listed.text
        assert [group["id"] for group in listed.json()["data"]] == [group_id]
        updated = clients.test_client.patch(
            f"{path}/{group_id}", headers=delegated_admin, json={"description": "checked"}
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["description"] == "checked"
        assert clients.test_client.post(path, headers=denied, json=submission).status_code == 403

        assert clients.access_log is not None
        entity_path = f"/apis/entities/v2/workspaces/{workspace}/entities/platform_job_group"
        writes = [
            request
            for request in clients.access_log.filter(path_startswith=entity_path)
            if request.method in {"POST", "PUT"}
        ]
        assert [request.method for request in writes] == ["POST", "PUT"]
        assert all(request.principal_id == "service:jobs" for request in writes)
        assert all(request.on_behalf_of == TEST_ADMIN_EMAIL for request in writes)

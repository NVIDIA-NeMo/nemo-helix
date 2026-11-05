# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Queued Builder submissions persist their creator using ordinary service credentials."""

import pytest
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.controller import _submitter
from nemo_builder_plugin.entities import QueuedBuild
from nemo_builder_plugin.service import BuilderService
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.entities import SyncEntityClient
from nemo_helix_plugin.entities.client import EntitiesClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.execution_profiles import KubernetesJobExecutionProfile
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceMemberRequest, CreateWorkspaceRequest
from nhx.common.auth.models import Principal
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
from nhx.testing import TEST_ADMIN_EMAIL, as_user, create_test_client, short_unique_name

pytestmark = pytest.mark.integration


def test_queued_build_preserves_creator_and_uses_builder_credentials(monkeypatch) -> None:
    config = BuilderConfig(
        registry="registry.invalid",
        sandbox={"image": "kaniko.invalid/executor:debug"},
        max_concurrent_builds=1,
        signing_key_secret=None,
        registry_username_secret=None,
        registry_password_secret=None,
    )
    monkeypatch.setattr(BuilderConfig, "get", lambda: config)

    async def profiles(self):
        return [
            KubernetesJobExecutionProfile.model_validate(
                {"profile": name, "config": {"namespace": "builds", "storage": {"pvc_name": "work"}}}
            )
            for name in ("build-fetch", "build-control", "build-push")
        ]

    monkeypatch.setattr(AsyncJobsClient, "list_execution_profiles", profiles)
    with create_test_client(
        lambda: NemoServiceAdapter(BuilderService()), auth_enabled=True, client_type=NemoClient
    ) as base:
        workspace = short_unique_name("builder-creator")
        admin = WorkspacesClient.from_client(as_user(base, TEST_ADMIN_EMAIL))
        admin.create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        admin.create_workspace_member(
            workspace=workspace,
            body=CreateWorkspaceMemberRequest(principal="builders", roles=["Editor"]),
            query_params={"wait_role_propagation": True},
        ).data()
        identity = Principal(
            id="service:integration-test",
            on_behalf_of="creator",
            on_behalf_of_email="creator@example.com",
            on_behalf_of_groups=["builders"],
            on_behalf_of_account_id="creator-account",
            on_behalf_of_authz_aliases=["creator-alias"],
        )
        caller = base.with_headers(identity.get_headers())
        response = caller._http.post(
            f"/apis/builder/v2/workspaces/{workspace}/builds",
            headers=caller.default_headers,
            json={
                "name": "queued",
                "revision": 1,
                "build_specs": [
                    {
                        "name": "image",
                        "source": {"fileset": "source"},
                        "output": {"repository": "team/image", "tag": "v1"},
                    }
                ],
            },
        )
        assert response.status_code == 201, response.text
        name = response.json()["job"]
        service = EntitiesClient.from_client(as_user(base, "service:builder"))
        stored = service.get_entity_by_name(workspace=workspace, entity_type="builder_queued_build", name=name).data()
        assert stored.created_by == "service:builder"
        snapshot = stored.data["_auth_context"]
        assert snapshot["principal_on_behalf_of"] == "creator"
        assert snapshot["principal_on_behalf_of_email"] == "creator@example.com"
        assert snapshot["principal_on_behalf_of_groups"] == ["builders"]
        assert snapshot["principal_on_behalf_of_account_id"] == "creator-account"
        assert snapshot["principal_on_behalf_of_authz_aliases"] == ["creator-alias"]
        build = SyncEntityClient(service).get(QueuedBuild, name=name, workspace=workspace)
        submitter = _submitter(build)
        assert submitter is not None and not isinstance(submitter, str)
        assert submitter.on_behalf_of == "creator"
        hidden = (
            EntitiesClient.from_client(caller)
            .get_entity_by_name(workspace=workspace, entity_type="builder_queued_build", name=name)
            .data()
        )
        assert "_auth_context" not in hidden.data

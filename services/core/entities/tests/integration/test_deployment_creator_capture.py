# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owning Deployments service captures OBO creators through real Entity Store CRUD."""

import pytest
from nemo_deployments_plugin.service import DeploymentsService
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.deployments.client import DeploymentsClient
from nemo_helix_plugin.deployments.types import CreateDeploymentConfigRequest, CreateDeploymentRequest
from nemo_helix_plugin.entities.client import EntitiesClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceMemberRequest, CreateWorkspaceRequest
from nhx.common.auth.models import Principal
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
from nhx.testing import TEST_ADMIN_EMAIL, as_user, create_test_client, short_unique_name

pytestmark = pytest.mark.integration


def test_deployment_captures_obo_creator_and_hides_snapshot() -> None:
    with create_test_client(
        lambda: NemoServiceAdapter(DeploymentsService()), auth_enabled=True, client_type=NemoClient
    ) as base:
        workspace = short_unique_name("deployment-obo")
        workspaces = WorkspacesClient.from_client(as_user(base, TEST_ADMIN_EMAIL))
        workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        workspaces.create_workspace_member(
            workspace=workspace,
            body=CreateWorkspaceMemberRequest(principal="creators", roles=["Editor"]),
            query_params={"wait_role_propagation": True},
        ).data()
        user = Principal(
            id="service:integration-test",
            on_behalf_of="creator@example.com",
            on_behalf_of_email="creator@example.com",
            on_behalf_of_groups=["creators"],
        )
        caller = base.with_headers(user.get_headers())
        deployments = DeploymentsClient.from_client(caller)
        deployments.create_deployment_config(
            workspace=workspace,
            body=CreateDeploymentConfigRequest(name="config"),
        ).data()
        created = deployments.create_deployment(
            workspace=workspace,
            body=CreateDeploymentRequest(name="run", deployment_config="config"),
        ).data()
        assert created.auth_context is None
        assert deployments.get_deployment(workspace=workspace, name="run").data().auth_context is None
        direct = as_user(base, "service:deployments")
        stored = (
            EntitiesClient.from_client(direct)
            .get_entity_by_name(
                workspace=workspace,
                entity_type="deployments_deployment",
                name="run",
            )
            .data()
        )
        assert stored.created_by == "service:deployments"
        assert stored.data["_auth_context"]["principal_id"] == "service:integration-test"
        assert stored.data["_auth_context"]["principal_on_behalf_of"] == "creator@example.com"
        projected = DeploymentsClient.from_client(direct).get_deployment(workspace=workspace, name="run").data()
        assert projected.auth_context is not None
        assert projected.auth_context["principal_on_behalf_of_groups"] == ["creators"]

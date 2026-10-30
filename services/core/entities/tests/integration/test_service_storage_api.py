# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit storage bindings through real persistence, HTTP handlers and PDP."""

from collections.abc import Iterator

import pytest
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.entities.client import EntitiesClient
from nemo_helix_plugin.entities.types import EntityCreateInput, EntityUpdate
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import (
    CreateWorkspaceMemberRequest,
    CreateWorkspaceRequest,
    UpdateWorkspaceMemberRequest,
)
from nhx.core.entities.service import EntitiesService
from nhx.testing import ClientContext, create_test_client, short_unique_name

pytestmark = pytest.mark.integration
SERVICE = "service:files"


@pytest.fixture(scope="module")
def platform() -> Iterator[ClientContext]:
    with create_test_client(
        EntitiesService, auth_enabled=True, workspaces=[], projects=[], client_type=ClientContext
    ) as context:
        yield context


def test_service_creator_gets_admin_and_can_add_storage_permissions(platform: ClientContext) -> None:
    client = platform.client.with_headers({"X-NHX-Principal-Id": SERVICE})
    workspaces = WorkspacesClient.from_client(client)
    workspace = workspaces.create_workspace(body=CreateWorkspaceRequest(name=short_unique_name("service-store"))).data()
    assert workspaces.get_workspace(name=workspace.name).data().created_by == SERVICE
    infrastructure = platform.client.with_headers({"X-NHX-Principal-Id": "service:auth"})
    members = WorkspacesClient.from_client(infrastructure).list_workspace_members(workspace=workspace.name).data()
    assert next(member.roles for member in members.data if member.principal == SERVICE) == ["Admin"]

    entities = EntitiesClient.from_client(client)
    with pytest.raises(PermissionDeniedError):
        list(entities.list_entities(workspace=workspace.name, entity_type="operational_record").items())
    workspaces.create_workspace_member(
        workspace=workspace.name,
        body=CreateWorkspaceMemberRequest(principal=SERVICE, roles=["ServiceEntityStore"]),
    ).data()
    entity = entities.create_entity(
        workspace=workspace.name,
        entity_type="operational_record",
        body=EntityCreateInput(name="record", data={"state": "queued"}),
    ).data()
    assert entities.get_entity_by_id(entity_id=entity.id).data().workspace == workspace.name
    assert len(list(entities.list_entities(workspace=workspace.name, entity_type="operational_record").items())) == 1
    updated = entities.update_entity_by_name(
        workspace=workspace.name,
        entity_type="operational_record",
        name="record",
        body=EntityUpdate(data={"state": "done"}),
    ).data()
    assert updated.data == {"state": "done"}
    entities.delete_entity_by_name(workspace=workspace.name, entity_type="operational_record", name="record").data()
    assert list(entities.list_entities(workspace=workspace.name, entity_type="operational_record").items()) == []


def test_storage_and_metadata_permissions_remain_independent_over_http(platform: ClientContext) -> None:
    client = platform.client.with_headers({"X-NHX-Principal-Id": SERVICE})
    workspaces = WorkspacesClient.from_client(client)
    workspace = workspaces.create_workspace(body=CreateWorkspaceRequest(name=short_unique_name("role-split"))).data()
    entities = EntitiesClient.from_client(client)
    workspaces.create_workspace_member(
        workspace=workspace.name,
        body=CreateWorkspaceMemberRequest(principal=SERVICE, roles=["ServiceEntityStore"]),
    ).data()
    workspaces.create_workspace_member(
        workspace=workspace.name,
        body=CreateWorkspaceMemberRequest(principal="operator", roles=["Admin"]),
    ).data()
    entity = entities.create_entity(
        workspace=workspace.name, entity_type="operational_record", body=EntityCreateInput(name="record", data={})
    ).data()
    infrastructure = platform.client.with_headers({"X-NHX-Principal-Id": "service:auth"})
    management = WorkspacesClient.from_client(infrastructure)
    management.update_workspace_member(
        workspace=workspace.name, principal_id=SERVICE, body=UpdateWorkspaceMemberRequest(roles=["Viewer"])
    ).data()
    assert workspaces.get_workspace(name=workspace.name).data().name == workspace.name
    with pytest.raises(PermissionDeniedError):
        entities.get_entity_by_id(entity_id=entity.id).data()
    with pytest.raises(PermissionDeniedError):
        list(entities.list_entities(workspace=workspace.name, entity_type="operational_record").items())

    management.update_workspace_member(
        workspace=workspace.name, principal_id=SERVICE, body=UpdateWorkspaceMemberRequest(roles=["ServiceEntityStore"])
    ).data()
    assert entities.get_entity_by_id(entity_id=entity.id).data().name == "record"
    with pytest.raises(PermissionDeniedError):
        workspaces.get_workspace(name=workspace.name).data()


def test_storage_grants_cannot_access_control_data_or_manage_members(platform: ClientContext) -> None:
    client = platform.client.with_headers({"X-NHX-Principal-Id": SERVICE})
    workspaces = WorkspacesClient.from_client(client)
    workspace = workspaces.create_workspace(body=CreateWorkspaceRequest(name=short_unique_name("control-guard"))).data()
    entities = EntitiesClient.from_client(client)
    infrastructure = platform.client.with_headers({"X-NHX-Principal-Id": "service:auth"})
    management = EntitiesClient.from_client(infrastructure)
    workspaces.create_workspace_member(
        workspace=workspace.name,
        body=CreateWorkspaceMemberRequest(principal="operator", roles=["Admin"]),
    ).data()
    workspaces.update_workspace_member(
        workspace=workspace.name,
        principal_id=SERVICE,
        body=UpdateWorkspaceMemberRequest(roles=["ServiceEntityStore"]),
    ).data()
    for entity_type in ("role_binding", "access_key"):
        control = management.create_entity(
            workspace=workspace.name,
            entity_type=entity_type,
            body=EntityCreateInput(
                name="existing-control",
                data={
                    "principal": "ordinary-user",
                    "workspace": workspace.name,
                    "role": "Viewer",
                    "granted_by": "service:auth",
                    "granted_at": "2026-10-29T17:00:00",
                    "revoked_at": None,
                },
            ),
        ).data()
        with pytest.raises(PermissionDeniedError):
            entities.create_entity(
                workspace=workspace.name, entity_type=entity_type, body=EntityCreateInput(name="test", data={})
            ).data()
        with pytest.raises(PermissionDeniedError):
            list(entities.list_entities(workspace=workspace.name, entity_type=entity_type).items())
        with pytest.raises(PermissionDeniedError):
            entities.get_entity_by_name(workspace=workspace.name, entity_type=entity_type, name=control.name).data()
        with pytest.raises(PermissionDeniedError):
            entities.get_entity_by_id(entity_id=control.id).data()
        with pytest.raises(PermissionDeniedError):
            entities.update_entity_by_name(
                workspace=workspace.name,
                entity_type=entity_type,
                name=control.name,
                body=EntityUpdate(data={}),
            ).data()
        with pytest.raises(PermissionDeniedError):
            entities.delete_entity_by_name(workspace=workspace.name, entity_type=entity_type, name=control.name).data()
    with pytest.raises(PermissionDeniedError):
        workspaces.list_workspace_members(workspace=workspace.name).data()

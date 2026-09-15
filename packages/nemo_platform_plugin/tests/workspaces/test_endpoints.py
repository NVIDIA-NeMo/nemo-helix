# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Workspaces service endpoint definitions."""

from __future__ import annotations

import json
from typing import get_origin

from nemo_platform_plugin.client.types import Paginated, PreparedRequest
from nemo_platform_plugin.entities.types import DeleteResponse
from nemo_platform_plugin.workspaces import endpoints
from nemo_platform_plugin.workspaces.types import (
    CreateWorkspaceMemberRequest,
    CreateWorkspaceRequest,
    UpdateWorkspaceMemberRequest,
    UpdateWorkspaceRequest,
    Workspace,
    WorkspaceMember,
    WorkspaceMemberListResponse,
)


def _json_body(prepared: PreparedRequest) -> dict:
    """Decode a prepared request's JSON body (asserting it is present bytes)."""
    assert isinstance(prepared.content, bytes)
    return json.loads(prepared.content)


def test_get_workspace() -> None:
    prepared = endpoints.get_workspace(name="ml-team")

    assert isinstance(prepared, PreparedRequest)
    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/entities/v2/workspaces/{name}"
    assert prepared.path_params == {"name": "ml-team"}
    assert prepared.response_type is Workspace


def test_list_workspaces() -> None:
    prepared = endpoints.list_workspaces()

    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/entities/v2/workspaces"
    assert prepared.path_params == {}
    assert prepared.query_params is None
    assert get_origin(prepared.response_type) is Paginated


def test_list_workspaces_with_query_params() -> None:
    prepared = endpoints.list_workspaces(query_params={"page": 2, "page_size": 5, "sort": "-name", "filter": "x"})

    assert prepared.query_params == {"page": 2, "page_size": 5, "sort": "-name", "filter": "x"}


def test_create_workspace_serializes_only_set_fields() -> None:
    prepared = endpoints.create_workspace(body=CreateWorkspaceRequest(name="ml-team"))

    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/entities/v2/workspaces"
    assert prepared.content_type == "application/json"
    assert _json_body(prepared) == {"name": "ml-team"}
    assert prepared.query_params is None
    assert prepared.response_type is Workspace


def test_create_workspace_with_query_params_and_exist_ok() -> None:
    prepared = endpoints.create_workspace(
        body=CreateWorkspaceRequest(name="ml-team", description="d"),
        query_params={"wait_role_propagation": False},
        exist_ok=True,
    )

    assert _json_body(prepared) == {"name": "ml-team", "description": "d"}
    assert prepared.query_params == {"wait_role_propagation": False}
    assert prepared.client_options == {"exist_ok": True}
    assert prepared.on_conflict_get is not None
    assert prepared.on_conflict_get.path_params == {"name": "ml-team"}


def test_update_workspace() -> None:
    prepared = endpoints.update_workspace(name="ml-team", body=UpdateWorkspaceRequest(description="new"))

    assert prepared.method == "PUT"
    assert prepared.path_template == "/apis/entities/v2/workspaces/{name}"
    assert prepared.path_params == {"name": "ml-team"}
    assert _json_body(prepared) == {"description": "new"}
    assert prepared.response_type is Workspace


def test_delete_workspace() -> None:
    prepared = endpoints.delete_workspace(name="ml-team")

    assert prepared.method == "DELETE"
    assert prepared.path_params == {"name": "ml-team"}
    assert prepared.content is None
    assert prepared.response_type is DeleteResponse


def test_list_workspace_members() -> None:
    prepared = endpoints.list_workspace_members(workspace="ml-team")

    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/entities/v2/workspaces/{workspace}/members"
    assert prepared.path_params == {"workspace": "ml-team"}
    assert prepared.response_type is WorkspaceMemberListResponse


def test_list_workspace_members_workspace_optional() -> None:
    prepared = endpoints.list_workspace_members()

    assert prepared.path_params == {}


def test_create_workspace_member() -> None:
    body = CreateWorkspaceMemberRequest(principal="user@example.com", roles=["Viewer"])
    prepared = endpoints.create_workspace_member(workspace="ml-team", body=body)

    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/entities/v2/workspaces/{workspace}/members"
    assert prepared.path_params == {"workspace": "ml-team"}
    assert _json_body(prepared) == {"principal": "user@example.com", "roles": ["Viewer"]}
    assert prepared.query_params is None
    assert prepared.response_type is WorkspaceMember


def test_create_workspace_member_omits_default_roles() -> None:
    prepared = endpoints.create_workspace_member(body=CreateWorkspaceMemberRequest(principal="user@example.com"))

    assert prepared.path_params == {}
    assert _json_body(prepared) == {"principal": "user@example.com"}


def test_create_workspace_member_query_params() -> None:
    prepared = endpoints.create_workspace_member(
        workspace="ml-team",
        body=CreateWorkspaceMemberRequest(principal="user@example.com"),
        query_params={"wait_role_propagation": False},
    )

    assert prepared.query_params == {"wait_role_propagation": False}


def test_update_workspace_member() -> None:
    prepared = endpoints.update_workspace_member(
        workspace="ml-team",
        principal_id="user@example.com",
        body=UpdateWorkspaceMemberRequest(roles=["Viewer", "Editor"]),
        query_params={"wait_role_propagation": True},
    )

    assert prepared.method == "PUT"
    assert prepared.path_template == "/apis/entities/v2/workspaces/{workspace}/members/{principal_id}"
    assert prepared.path_params == {"workspace": "ml-team", "principal_id": "user@example.com"}
    assert _json_body(prepared) == {"roles": ["Viewer", "Editor"]}
    assert prepared.query_params == {"wait_role_propagation": True}
    assert prepared.response_type is WorkspaceMember


def test_delete_workspace_member() -> None:
    prepared = endpoints.delete_workspace_member(principal_id="user-123", query_params={"wait_role_propagation": False})

    assert prepared.method == "DELETE"
    assert prepared.path_template == "/apis/entities/v2/workspaces/{workspace}/members/{principal_id}"
    assert prepared.path_params == {"principal_id": "user-123"}
    assert prepared.query_params == {"wait_role_propagation": False}
    assert prepared.content is None
    assert prepared.response_type is DeleteResponse

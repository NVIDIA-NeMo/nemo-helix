# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for WorkspacesClient / AsyncWorkspacesClient over a recording httpx transport."""

from __future__ import annotations

import json

import httpx
import pytest
from nemo_platform_plugin.client.errors import ConflictError, NotFoundError
from nemo_platform_plugin.workspaces.client import AsyncWorkspacesClient, WorkspacesClient
from nemo_platform_plugin.workspaces.types import (
    CreateWorkspaceMemberRequest,
    CreateWorkspaceRequest,
    UpdateWorkspaceMemberRequest,
)

BASE = "http://test:8000"

WORKSPACE = {
    "id": "8a4d8f1e-0f7f-4b8a-9d5f-2f4b6c8e1a2b",
    "name": "ml-team",
    "description": None,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
}
MEMBER = {"principal": "user@example.com", "roles": ["Editor"], "granted_at": None, "granted_by": "system"}


class Recorder:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]


def make_client(recorder: Recorder, *, workspace: str | None = "default") -> WorkspacesClient:
    return WorkspacesClient(
        base_url=BASE, workspace=workspace, http_client=httpx.Client(transport=httpx.MockTransport(recorder))
    )


def test_create_workspace_sends_query_params() -> None:
    recorder = Recorder([httpx.Response(201, json=WORKSPACE)])
    client = make_client(recorder)

    workspace = client.create_workspace(
        body=CreateWorkspaceRequest(name="ml-team"), query_params={"wait_role_propagation": False}
    ).data()

    assert workspace.name == "ml-team"
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == "/apis/entities/v2/workspaces"
    assert dict(recorder.last.url.params) == {"wait_role_propagation": "false"}
    assert json.loads(recorder.last.content) == {"name": "ml-team"}


def test_create_workspace_exist_ok_replays_get_on_conflict() -> None:
    recorder = Recorder([httpx.Response(409, json={"detail": "exists"}), httpx.Response(200, json=WORKSPACE)])
    client = make_client(recorder)

    workspace = client.create_workspace(body=CreateWorkspaceRequest(name="ml-team"), exist_ok=True).data()

    assert workspace.name == "ml-team"
    assert [(r.method, r.url.path) for r in recorder.requests] == [
        ("POST", "/apis/entities/v2/workspaces"),
        ("GET", "/apis/entities/v2/workspaces/ml-team"),
    ]


def test_create_workspace_conflict_raises_without_exist_ok() -> None:
    recorder = Recorder([httpx.Response(409, json={"detail": "exists"})])
    client = make_client(recorder)

    with pytest.raises(ConflictError):
        client.create_workspace(body=CreateWorkspaceRequest(name="ml-team"))


def test_get_workspace_not_found_raises() -> None:
    recorder = Recorder([httpx.Response(404, json={"detail": "Workspace 'missing' not found"})])
    client = make_client(recorder)

    with pytest.raises(NotFoundError) as exc:
        client.get_workspace(name="missing")
    assert exc.value.status_code == 404


def test_list_workspace_members_uses_client_default_workspace() -> None:
    recorder = Recorder([httpx.Response(200, json={"data": [MEMBER]})])
    client = make_client(recorder)

    members = client.list_workspace_members().data()

    assert [m.principal for m in members.data] == ["user@example.com"]
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/default/members"


def test_list_workspace_members_without_workspace_raises() -> None:
    recorder = Recorder([])
    client = make_client(recorder, workspace=None)

    with pytest.raises(ValueError, match="Missing path parameter 'workspace'"):
        client.list_workspace_members()
    assert recorder.requests == []


def test_create_workspace_member_sends_body_and_query_params() -> None:
    recorder = Recorder([httpx.Response(201, json=MEMBER)])
    client = make_client(recorder)

    member = client.create_workspace_member(
        workspace="ml-team",
        body=CreateWorkspaceMemberRequest(principal="user@example.com", roles=["Editor"]),
        query_params={"wait_role_propagation": False},
    ).data()

    assert member.principal == "user@example.com"
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/ml-team/members"
    assert dict(recorder.last.url.params) == {"wait_role_propagation": "false"}
    assert json.loads(recorder.last.content) == {"principal": "user@example.com", "roles": ["Editor"]}


def test_update_workspace_member_encodes_principal_in_path() -> None:
    recorder = Recorder([httpx.Response(200, json={**MEMBER, "roles": ["Viewer"]})])
    client = make_client(recorder)

    member = client.update_workspace_member(
        principal_id="user@example.com", body=UpdateWorkspaceMemberRequest(roles=["Viewer"])
    ).data()

    assert member.roles == ["Viewer"]
    assert recorder.last.method == "PUT"
    assert recorder.last.url.raw_path == b"/apis/entities/v2/workspaces/default/members/user%40example.com"
    assert dict(recorder.last.url.params) == {}
    assert json.loads(recorder.last.content) == {"roles": ["Viewer"]}


def test_delete_workspace_member_sends_query_params() -> None:
    recorder = Recorder([httpx.Response(200, json={"message": "deleted", "id": "user-123", "deleted_at": None})])
    client = make_client(recorder)

    client.delete_workspace_member(
        workspace="ml-team", principal_id="user-123", query_params={"wait_role_propagation": True}
    )

    assert recorder.last.method == "DELETE"
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/ml-team/members/user-123"
    assert dict(recorder.last.url.params) == {"wait_role_propagation": "true"}


async def test_async_client_create_member() -> None:
    recorder = Recorder([httpx.Response(201, json=MEMBER)])
    client = AsyncWorkspacesClient(
        base_url=BASE, workspace="default", http_client=httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    )

    response = await client.create_workspace_member(
        body=CreateWorkspaceMemberRequest(principal="user@example.com"), query_params={"wait_role_propagation": False}
    )

    assert response.data().principal == "user@example.com"
    assert recorder.last.url.path == "/apis/entities/v2/workspaces/default/members"
    assert dict(recorder.last.url.params) == {"wait_role_propagation": "false"}
    assert json.loads(recorder.last.content) == {"principal": "user@example.com"}

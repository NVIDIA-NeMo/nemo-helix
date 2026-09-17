# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Workspaces API (Entity Store).

Single source of truth for the HTTP contract. Replaces the Stainless-generated
``nemo_platform.resources.workspaces`` resource.
"""

from __future__ import annotations

from abc import abstractmethod

from nemo_platform_plugin.client.endpoint import delete, get, post, put
from nemo_platform_plugin.client.types import Paginated, PreparedRequest
from nemo_platform_plugin.entities.types import DeleteResponse
from nemo_platform_plugin.workspaces.types import (
    CreateWorkspaceMemberRequest,
    CreateWorkspaceQueryParams,
    CreateWorkspaceRequest,
    ListWorkspacesQueryParams,
    UpdateWorkspaceMemberRequest,
    UpdateWorkspaceRequest,
    Workspace,
    WorkspaceMember,
    WorkspaceMemberListResponse,
    WorkspaceMemberQueryParams,
)

# ---------------------------------------------------------------------------
# Workspace CRUD
# ---------------------------------------------------------------------------


@get("/apis/entities/v2/workspaces/{name}")
@abstractmethod
def get_workspace(*, name: str) -> Workspace: ...


@get("/apis/entities/v2/workspaces")
@abstractmethod
def list_workspaces(*, query_params: ListWorkspacesQueryParams | None = None) -> Paginated[Workspace]: ...


def _get_workspace_on_conflict(body: CreateWorkspaceRequest, workspace: str | None) -> PreparedRequest[Workspace]:
    """Build the retrieve request replayed when ``create_workspace(exist_ok=True)`` 409s."""
    return get_workspace(name=body.name)


@post("/apis/entities/v2/workspaces", get_on_conflict=_get_workspace_on_conflict)
@abstractmethod
def create_workspace(
    *,
    body: CreateWorkspaceRequest,
    query_params: CreateWorkspaceQueryParams | None = None,
    exist_ok: bool = False,
) -> Workspace: ...


@put("/apis/entities/v2/workspaces/{name}")
@abstractmethod
def update_workspace(*, name: str, body: UpdateWorkspaceRequest) -> Workspace: ...


@delete("/apis/entities/v2/workspaces/{name}")
@abstractmethod
def delete_workspace(*, name: str) -> DeleteResponse: ...


# ---------------------------------------------------------------------------
# Workspace members
# ---------------------------------------------------------------------------


@get("/apis/entities/v2/workspaces/{workspace}/members")
@abstractmethod
def list_workspace_members(*, workspace: str | None = None) -> WorkspaceMemberListResponse: ...


@post("/apis/entities/v2/workspaces/{workspace}/members")
@abstractmethod
def create_workspace_member(
    *,
    workspace: str | None = None,
    body: CreateWorkspaceMemberRequest,
    query_params: WorkspaceMemberQueryParams | None = None,
) -> WorkspaceMember: ...


@put("/apis/entities/v2/workspaces/{workspace}/members/{principal_id}")
@abstractmethod
def update_workspace_member(
    *,
    workspace: str | None = None,
    principal_id: str,
    body: UpdateWorkspaceMemberRequest,
    query_params: WorkspaceMemberQueryParams | None = None,
) -> WorkspaceMember: ...


@delete("/apis/entities/v2/workspaces/{workspace}/members/{principal_id}")
@abstractmethod
def delete_workspace_member(
    *, workspace: str | None = None, principal_id: str, query_params: WorkspaceMemberQueryParams | None = None
) -> DeleteResponse: ...

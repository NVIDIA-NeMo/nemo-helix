# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request/response types for the Workspaces API (Entity Store).

One source of truth for the HTTP contract, used by both the server routes and
the typed NemoClient endpoints. Replaces the Stainless-generated
``nemo_platform.types.workspaces`` module.
"""

from __future__ import annotations

from datetime import datetime
from typing import NotRequired, TypedDict

from nemo_platform_plugin.schema import Page
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


class Workspace(BaseModel):
    """Workspace schema for API responses."""

    id: str = Field(description="System-generated UUID")
    name: str = Field(description="Workspace name (user-provided)")
    description: str | None = None
    created_at: datetime
    created_by: str | None = None
    updated_at: datetime
    updated_by: str | None = None


WorkspacePage = Page[Workspace]


class WorkspaceMember(BaseModel):
    """Workspace member response model."""

    principal: str
    roles: list[str]
    granted_at: datetime | None = None
    granted_by: str | None = None


class WorkspaceMemberListResponse(BaseModel):
    data: list[WorkspaceMember]


# ---------------------------------------------------------------------------
# Request types
# ---------------------------------------------------------------------------


class CreateWorkspaceRequest(BaseModel):
    """Schema for creating a new workspace."""

    name: str
    description: str | None = None


class UpdateWorkspaceRequest(BaseModel):
    """Schema for updating a workspace."""

    description: str | None = None


class CreateWorkspaceMemberRequest(BaseModel):
    """Schema for adding a new workspace member."""

    principal: str
    roles: list[str] = Field(default_factory=lambda: ["Editor"])


class UpdateWorkspaceMemberRequest(BaseModel):
    """Schema for updating a workspace member's roles."""

    roles: list[str]


# ---------------------------------------------------------------------------
# Query parameter types
# ---------------------------------------------------------------------------


class ListWorkspacesQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]


class CreateWorkspaceQueryParams(TypedDict, total=False):
    wait_role_propagation: NotRequired[bool]


class WorkspaceMemberQueryParams(TypedDict, total=False):
    """Query parameters shared by the member create/update/delete endpoints."""

    wait_role_propagation: NotRequired[bool]

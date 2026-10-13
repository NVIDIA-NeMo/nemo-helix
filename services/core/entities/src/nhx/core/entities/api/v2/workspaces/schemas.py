# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workspace API Schemas for v2."""

from datetime import datetime
from typing import List, Optional

from nhx.common.entities.constants import NAME_PATTERN, NAME_PATTERN_DESCRIPTION
from pydantic import BaseModel, ConfigDict, Field


class WorkspaceInput(BaseModel):
    """Schema for creating a new workspace."""

    name: str = Field(
        ...,
        description=f"Workspace name (unique identifier). {NAME_PATTERN_DESCRIPTION}",
        pattern=NAME_PATTERN,
        examples=["ml-team", "research"],
    )
    description: Optional[str] = Field(None, description="Optional description of the workspace")

    model_config = ConfigDict(regex_engine="python-re")


class WorkspaceUpdate(BaseModel):
    """Schema for updating a workspace."""

    description: Optional[str] = Field(None, description="Updated description")


# =================== Workspace Member Schemas ===================


class WorkspaceMemberInput(BaseModel):
    """Schema for adding a new workspace member."""

    principal: str = Field(
        ...,
        description="The principal identifier (email, user ID, or group ID)",
        examples=["user@example.com", "user-123"],
    )
    roles: List[str] = Field(
        default=["Editor"],
        description="List of roles to grant to the principal",
        examples=[["Editor"], ["Viewer", "Editor"]],
    )


class WorkspaceMemberUpdate(BaseModel):
    """Schema for updating a workspace member's roles."""

    roles: List[str] = Field(
        ...,
        description="Updated list of roles for the principal",
        examples=[["Viewer"], ["Editor", "Admin"]],
    )


class WorkspaceMember(BaseModel):
    """Workspace member response model."""

    principal: str = Field(..., description="The principal identifier")
    roles: List[str] = Field(..., description="List of roles granted to the principal")
    granted_at: Optional[datetime] = Field(None, description="Earliest date when a role was granted to this principal")
    granted_by: Optional[str] = Field(None, description="Who granted the earliest role")


class WorkspaceMemberListResponse(BaseModel):
    """Schema for workspace member list responses."""

    data: List[WorkspaceMember]

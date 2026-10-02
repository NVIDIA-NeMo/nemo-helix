# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IAM schemas for API endpoints.

Role-binding request and response DTOs are shared with the typed plugin client.
Server-only filter schemas remain here because they provide entity-store field
translation that is not part of the wire DTO contract.
"""

from datetime import datetime
from typing import Optional

from nemo_helix_plugin.iam.types import RoleBinding as RoleBinding
from nemo_helix_plugin.iam.types import RoleBindingInput as RoleBindingInput
from nhx.common.entities.values import Filter, StringFilter
from pydantic import BaseModel, Field


class DateRangeFilter(BaseModel):
    """Filter for date ranges."""

    gte: Optional[datetime] = Field(None, description="Greater than or equal to this date")
    lte: Optional[datetime] = Field(None, description="Less than or equal to this date")


class RoleBindingUpdate(BaseModel):
    """Input schema for updating a role binding."""

    revoked_at: Optional[datetime] = Field(
        default=None, description="Timestamp when the role was revoked (None if active)"
    )


class RoleBindingFilter(Filter):
    """Filter for role bindings."""

    principal: StringFilter | str | None = Field(None, description="Filter by principal ID")
    workspace: Optional[str] = Field(None, description="Filter by workspace")
    role: StringFilter | str | None = Field(None, description="Filter by role")
    granted_by: StringFilter | str | None = Field(None, description="Filter by who granted the role")
    is_active: Optional[bool] = Field(default=None, description="Filter for active (True) or revoked (False) bindings")
    granted_at: Optional[DateRangeFilter] = Field(None, description="Filter by granted date range")
    revoked_at: Optional[DateRangeFilter] = Field(None, description="Filter by revoked date range")


class WorkspaceMemberInput(BaseModel):
    """Input schema for adding a workspace member."""

    principal: str = Field(description="The principal identifier (email, user ID, or group ID)")
    roles: list[str] = Field(default=["Editor"], description="List of roles to grant to the principal")


class WorkspaceMemberUpdate(BaseModel):
    """Input schema for updating a workspace member."""

    roles: list[str] = Field(description="Updated list of roles for the principal")


class WorkspaceMember(BaseModel):
    """Workspace member response model."""

    principal: str
    roles: list[str]

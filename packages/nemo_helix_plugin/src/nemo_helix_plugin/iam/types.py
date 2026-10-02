# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request and response types for the Auth service IAM API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, NotRequired, TypedDict

from pydantic import BaseModel, Field


class RoleBindingInput(BaseModel):
    """Input schema for creating a role binding."""

    principal: str = Field(description="The principal identifier (email, user ID, or group ID)")
    workspace: str | None = Field(
        default=None, description="The workspace this binding applies to. None for platform-level roles."
    )
    role: str = Field(description="The role name (e.g., 'Viewer', 'Editor', 'Admin')")


class RoleBinding(BaseModel):
    """Role binding response model."""

    id: str
    name: str
    principal: str
    workspace: str | None
    role: str
    granted_by: str
    granted_at: datetime
    revoked_at: datetime | None


class RoleBindingDeleteResponse(BaseModel):
    message: str = Field(default="Resource deleted successfully.")
    id: str | None = Field(default=None, description="The ID of the deleted resource.")
    deleted_at: datetime | None = Field(default=None, description="The timestamp when the resource was deleted.")


class AuthzRequest(BaseModel):
    """Authorization request input."""

    input: dict[str, Any] = Field(
        ...,
        description="Input data for policy evaluation",
        json_schema_extra={
            "examples": [
                {
                    "principal_id": "user@example.com",
                    "method": "GET",
                    "path": "/v2/workspaces/my-workspace/models",
                }
            ]
        },
    )


class AuthzResponse(BaseModel):
    """Authorization response."""

    result: dict[str, Any] = Field(..., description="Policy evaluation result")


class AuthzErrorDetail(BaseModel):
    """Structured detail returned for an invalid authorization entrypoint."""

    error: str = Field(..., description="Error message")
    valid_entrypoints: list[str] = Field(default_factory=list, description="List of valid entrypoints")


class AuthzErrorResponse(BaseModel):
    """Invalid-entrypoint error envelope returned by authorization evaluation."""

    detail: AuthzErrorDetail


ListRoleBindingsQueryParams = TypedDict(
    "ListRoleBindingsQueryParams",
    {
        "page": NotRequired[int],
        "page_size": NotRequired[int],
        "sort": NotRequired[str],
        "filter": NotRequired[str],
        "filter[principal]": NotRequired[str],
        "filter[principal][$eq]": NotRequired[str],
        "filter[principal][$like]": NotRequired[str],
        "filter[principal][$in]": NotRequired[str],
        "filter[principal][$nin]": NotRequired[str],
        "filter[workspace]": NotRequired[str],
        "filter[role]": NotRequired[str],
        "filter[role][$eq]": NotRequired[str],
        "filter[role][$like]": NotRequired[str],
        "filter[role][$in]": NotRequired[str],
        "filter[role][$nin]": NotRequired[str],
        "filter[granted_by]": NotRequired[str],
        "filter[granted_by][$eq]": NotRequired[str],
        "filter[granted_by][$like]": NotRequired[str],
        "filter[granted_by][$in]": NotRequired[str],
        "filter[granted_by][$nin]": NotRequired[str],
        "filter[is_active]": NotRequired[bool],
        "filter[granted_at][$gte]": NotRequired[str],
        "filter[granted_at][$lte]": NotRequired[str],
        "filter[revoked_at][$gte]": NotRequired[str],
        "filter[revoked_at][$lte]": NotRequired[str],
    },
    total=False,
)


class RolePropagationQueryParams(TypedDict, total=False):
    wait_role_propagation: NotRequired[bool]

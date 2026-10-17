# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

AccessKeyStatus = Literal["ACTIVE", "EXPIRED", "REVOKED", "SUSPENDED", "ROTATING"]
AccessKeyReversibleStatus = Literal["ACTIVE", "EXPIRED", "SUSPENDED"]
AccessKeyEntityType = Literal["USER", "SERVICE_ACCOUNT"]
ACCESS_KEY_JTI_PATTERN = r"^ak_[0-9a-f]{32}$"
# Not blank once trimmed. Keep in sync with the `.strip()` checks below (this is what OpenAPI shows).
_NON_BLANK_PATTERN = r"^\s*\S.*$"
# A service/role name with no whitespace or ':' once trimmed (those chars are claim delimiters).
_SCOPE_ITEM_PATTERN = r"^\s*[^\s:]+\s*$"


class AccessKeyWorkspaceGrant(BaseModel):
    """Workspace membership to grant to a newly created access key principal."""

    workspace: str = Field(
        pattern=_NON_BLANK_PATTERN,
        description="Workspace name. Must not be blank.",
    )
    roles: list[str] = Field(
        default_factory=lambda: ["Editor"],
        json_schema_extra={
            "x-schema-default": ["Editor"],
            # Blank entries get dropped below, but at least one non-blank role must remain.
            "contains": {"pattern": _NON_BLANK_PATTERN},
            "minContains": 1,
        },
        description="Roles to grant in the workspace. Defaults to ['Editor'] when omitted.",
    )

    @model_validator(mode="after")
    def _normalize(self) -> Self:
        workspace = self.workspace.strip()
        if not workspace:
            raise ValueError("workspace must not be blank")
        # Reject empty roles instead of silently creating a no-op grant.
        roles = [role.strip() for role in self.roles if role.strip()]
        if not roles:
            raise ValueError("roles, if provided, must contain at least one non-empty role")
        self.workspace = workspace
        self.roles = roles
        return self


class AccessKeyBoundWorkspace(AccessKeyWorkspaceGrant):
    """A workspace a service-bound key is bound to, with the roles its service account gets there.

    Unlike a plain grant, the roles are required: binding a key to a workspace never grants a
    role implicitly.
    """

    roles: list[str] = Field(
        json_schema_extra={
            # Blank entries get dropped by the base class, but at least one non-blank role must remain.
            "contains": {"pattern": _NON_BLANK_PATTERN},
            "minContains": 1,
        },
        description="Roles to grant the key's service account in the workspace. Required.",
    )


class AccessKeyListQueryParams(TypedDict, total=False):
    """Pagination parameters for Scoped Access Key listing."""

    page: int
    page_size: int


class AccessKeyCreateRequest(BaseModel):
    """Request body for creating a Scoped Access Key."""

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        json_schema_extra={"nullable": True},
        description="Optional human-readable Scoped Access Key label. The token jti remains the stable identifier.",
    )
    description: str | None = Field(
        default=None,
        max_length=1024,
        json_schema_extra={"nullable": True},
        description="Optional human-readable description of the Scoped Access Key.",
    )
    expires_in_seconds: int | None = Field(
        default=None,
        ge=1,
        json_schema_extra={"nullable": True},
        description=(
            "Scoped Access Key lifetime in seconds. Omit to use "
            "auth.access_keys.default_expires_in_seconds. Send explicit null to request "
            "a non-time-delimited key, which requires auth.access_keys.max_expires_in_seconds "
            "to be disabled."
        ),
    )
    service_account_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=240,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._+/-]*$",
        json_schema_extra={"nullable": True},
        description=(
            "Optional non-human service account to bind the key to. Service-bound keys can be "
            "created by a HelixAdmin, or by an Admin of every workspace named in `bound_workspaces` for a "
            "service account named `<workspace>/<name>` after one of them, and authenticate as "
            "service-account:<id>."
        ),
    )
    bound_workspaces: list[AccessKeyBoundWorkspace] | None = Field(
        default=None,
        min_length=1,
        json_schema_extra={"nullable": True},
        description=(
            "Workspaces to bind a service-bound key to, each with the roles the key's service account "
            "gets there (no role is granted implicitly). Requires service_account_id and cannot be "
            "combined with `workspaces`. A key bound to several workspaces is managed by someone who is "
            "Admin of all of them, or by a HelixAdmin. A caller who is not a HelixAdmin must be Admin of "
            "every bound workspace, must name the service account `<workspace>/<name>` after one of them, "
            "and can only use a service account whose existing keys and role bindings are all within these "
            "workspaces (checked when the key is created or rotated, not continuously). The key is bound "
            "to each workspace's ID, so if a bound workspace is deleted and recreated under the same name, "
            "only a HelixAdmin can manage the key."
        ),
    )
    acknowledge_workspace_bound_keys: bool = Field(
        default=False,
        description=(
            "Confirm creating a service-bound key for an account that already has live keys bound to "
            "workspaces, when the new key grants roles in a workspace outside those a live key is bound "
            "to. Roles belong to the service account, so those keys gain the new roles too, while being "
            "managed by the bound workspaces' Admins. Without it, such a request fails with 409 and code "
            "`workspace_bound_keys_exist`."
        ),
    )
    scope: list[str] | None = Field(
        default=None,
        min_length=1,
        json_schema_extra={"nullable": True, "items": {"type": "string", "pattern": _SCOPE_ITEM_PATTERN}},
        description="Optional service names that restrict this key to read and write access for those services.",
    )
    rotates: str | None = Field(
        default=None,
        pattern=ACCESS_KEY_JTI_PATTERN,
        json_schema_extra={"nullable": True},
        description=(
            "JTI of a prior Scoped Access Key owned by the caller to revoke after creation. "
            "Intended primarily for personal keys without a service-account identity."
        ),
    )
    workspaces: list[AccessKeyWorkspaceGrant] | None = Field(
        default=None,
        json_schema_extra={"nullable": True},
        description="Optional workspace memberships to grant to the newly created key principal.",
    )

    @model_validator(mode="after")
    def _validate_scope(self) -> Self:
        if self.scope is None:
            return self
        # Reject empty scope instead of silently minting an unscoped, full-access token.
        normalized = [service.strip() for service in self.scope]
        if not normalized or any(not service for service in normalized):
            raise ValueError("scope, if provided, must contain at least one non-empty service name")
        # Service names later join into one space-delimited claim, so ':' or whitespace would corrupt it.
        if any(char.isspace() or char == ":" for service in normalized for char in service):
            raise ValueError("scope service names must not contain whitespace or ':' characters")
        self.scope = normalized
        return self

    @model_validator(mode="after")
    def _validate_workspace_binding(self) -> Self:
        """Require a service account for a bound key, and take its grants only from the bound workspaces."""
        if self.bound_workspaces is None:
            return self
        if self.service_account_id is None:
            raise ValueError("bound_workspaces requires service_account_id")
        if self.workspaces:
            raise ValueError("workspaces cannot be combined with bound_workspaces; grant roles in bound_workspaces")
        names = [bound.workspace for bound in self.bound_workspaces]
        if len(set(names)) != len(names):
            raise ValueError("bound_workspaces must not contain duplicate workspace names")
        return self

    @model_validator(mode="after")
    def _validate_workspace_grants(self) -> Self:
        seen: set[str] = set()
        for grant in self.workspaces or []:
            if grant.workspace in seen:
                raise ValueError(f"workspaces must not contain duplicate workspace names: {grant.workspace}")
            seen.add(grant.workspace)
        return self


class AccessKeyRotateRequest(BaseModel):
    """Request body for rotating a Scoped Access Key."""

    grace_period_seconds: int | None = Field(
        default=None,
        ge=1,
        json_schema_extra={"nullable": True},
        description=(
            "Grace period in seconds for the rotated-out key. Omit to use "
            "auth.access_keys.rotation_grace_period_seconds. Subject to "
            "auth.access_keys.max_rotation_grace_period_seconds."
        ),
    )


class AccessKeyMetadataResponse(BaseModel):
    """Metadata for a Scoped Access Key."""

    jti: str = Field(description="Stable JWT ID for this Scoped Access Key.")
    name: str | None = Field(
        default=None,
        json_schema_extra={"nullable": True},
        description="Optional human-readable Scoped Access Key label.",
    )
    description: str | None = Field(
        default=None,
        json_schema_extra={"nullable": True},
        description="Human-readable description of the Scoped Access Key.",
    )
    principal: str = Field(description="Principal ID stamped into the token.")
    entity_type: AccessKeyEntityType = Field(
        default="USER",
        description="Whether the key is bound to a user or a non-human service account.",
    )
    bound_workspaces: list[str] = Field(
        default_factory=list,
        description=(
            "Current names of the workspaces a service-bound key is bound to, aligned with "
            "`bound_workspace_ids`; a workspace that no longer exists, or that the caller can't see, is "
            "shown by its ID. Empty for personal and unbound keys."
        ),
    )
    bound_workspace_ids: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of the workspaces a service-bound key is bound to. The key stays bound to these IDs: a "
            "workspace deleted and recreated under the same name has a new ID and doesn't manage the key."
        ),
    )
    status: AccessKeyStatus
    issuer: str = Field(description="Issuer stamped into the Scoped Access Key JWT.")
    audiences: list[str] = Field(
        description="Audiences accepted for the Scoped Access Key JWT.",
        json_schema_extra={"uniqueItems": True},
    )
    scope: list[str] = Field(
        default_factory=list,
        description="Services this key is restricted to. An empty list means the key is unscoped.",
    )
    created_at: datetime
    expires_at: datetime | None = Field(default=None, json_schema_extra={"nullable": True})
    grace_period_expires_at: datetime | None = Field(
        default=None,
        json_schema_extra={"nullable": True},
        description="Timestamp when the rotated-out key's grace period expires.",
    )
    last_used_at: datetime | None = Field(
        default=None,
        json_schema_extra={"nullable": True},
        description="Timestamp of the most recent successful authentication with this Scoped Access Key.",
    )


class AccessKeyCreateResponse(AccessKeyMetadataResponse):
    """Create response. The token value is returned only once."""

    token: str
    token_type: Literal["Bearer"]


class AccessKeyListResponse(BaseModel):
    """List response for Scoped Access Key metadata."""

    data: list[AccessKeyMetadataResponse]
    has_more: bool = Field(
        default=False,
        description="True when another page of keys is available.",
    )


class AccessKeyRevokeResponse(BaseModel):
    """Response returned after a Scoped Access Key revoke request."""

    jti: str = Field(description="Stable JWT ID for this Scoped Access Key.")
    revoked: bool = Field(description="True when this request newly recorded the key's revocation.")


class AccessKeyStatusChangeResponse(BaseModel):
    """Response returned after a reversible Scoped Access Key status change."""

    jti: str = Field(description="Stable JWT ID for this Scoped Access Key.")
    status: AccessKeyReversibleStatus = Field(
        description="Resulting effective status of the key, including expiration."
    )
    changed: bool = Field(description="True when this request changed the key's persistent status.")


class AccessKeyRotateResponse(BaseModel):
    """Response returned after rotating a Scoped Access Key.

    The rotated-out key (``previous_jti``) remains usable for ``grace_period_seconds``
    (the dual-active grace period) so callers can cut traffic over to ``new_key``
    before the old key is treated as revoked.
    """

    new_key: AccessKeyCreateResponse = Field(
        description="Newly minted successor Scoped Access Key. Its raw token is returned only once."
    )
    previous_jti: str = Field(description="Stable JWT ID of the Scoped Access Key that was rotated out.")
    previous_status: AccessKeyStatus = Field(
        description=(
            "Effective status of the rotated-out key immediately after this request. Normally "
            "ROTATING, but may already read as REVOKED or EXPIRED if reconciling this request's "
            "outcome was itself delayed past the grace deadline or a concurrent revoke."
        )
    )
    grace_period_seconds: int = Field(
        description="Seconds the rotated-out key remains usable before it is treated as revoked."
    )
    grace_period_expires_at: datetime | None = Field(
        default=None,
        json_schema_extra={"nullable": True},
        description="Timestamp when the rotated-out key's grace period expires.",
    )


class AccessKeyNotImplementedErrorResponse(BaseModel):
    """Response returned by unsupported Scoped Access Key lifecycle endpoints."""

    detail: str


class AccessKeyErrorResponse(BaseModel):
    """Scoped Access Key error response."""

    detail: str
    code: Literal["access_keys_disabled", "workspace_bound_keys_exist"] | None = Field(
        default=None,
        json_schema_extra={"nullable": True},
        description=(
            "Set to access_keys_disabled when the Scoped Access Key feature is disabled, or to "
            "workspace_bound_keys_exist when creating the key needs acknowledge_workspace_bound_keys."
        ),
    )


class JsonWebKey(BaseModel):
    """JSON Web Key object."""

    model_config = ConfigDict(extra="allow")

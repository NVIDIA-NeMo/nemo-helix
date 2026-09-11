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


class AccessKeyWorkspaceGrant(BaseModel):
    """Workspace membership to grant to a newly created access key principal."""

    workspace: str
    roles: list[str] = Field(default_factory=lambda: ["Editor"])

    @model_validator(mode="after")
    def _normalize(self) -> Self:
        workspace = self.workspace.strip()
        if not workspace:
            raise ValueError("workspace must not be blank")
        # An explicit empty/blank-only roles list would otherwise reach the workspace-members
        # API as a no-op grant (no exception, no binding created) — reject it here instead of
        # silently creating a key whose requested access was never provisioned.
        roles = [role.strip() for role in self.roles if role.strip()]
        if not roles:
            raise ValueError("roles, if provided, must contain at least one non-empty role")
        self.workspace = workspace
        self.roles = roles
        return self


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
            "Optional non-human service account to bind the key to. Service-bound keys can only be "
            "created by a PlatformAdmin and authenticate as service-account:<id>."
        ),
    )
    scope: list[str] | None = Field(
        default=None,
        json_schema_extra={"nullable": True},
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
        # An explicitly empty (or whitespace/comma-only) scope would otherwise mint the same
        # unscoped, full-access token as omitting `scope` entirely — the opposite of what a
        # caller asking to restrict a key would expect. Fail closed instead.
        normalized = [service.strip() for service in self.scope]
        if not normalized or any(not service for service in normalized):
            raise ValueError("scope, if provided, must contain at least one non-empty service name")
        # Service names are later joined into a single space-delimited `service:read
        # service:write ...` claim, so a name containing whitespace or `:` would corrupt that
        # claim's delimiters and could be reinterpreted downstream as a different service's
        # scope entirely.
        if any(char.isspace() or char == ":" for service in normalized for char in service):
            raise ValueError("scope service names must not contain whitespace or ':' characters")
        self.scope = normalized
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
    code: Literal["access_keys_disabled"] | None = Field(
        default=None,
        json_schema_extra={"nullable": True},
        description="Set to access_keys_disabled when the Scoped Access Key feature is disabled.",
    )


class JsonWebKey(BaseModel):
    """JSON Web Key object."""

    model_config = ConfigDict(extra="allow")

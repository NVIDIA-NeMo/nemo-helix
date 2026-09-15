# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request/response types for the Secrets service.

These types define the HTTP contract for secret CRUD, value access, and the
admin key-rotation endpoint. Both the server (FastAPI routes) and the client
(NemoClient endpoints) import from here — one source of truth, no
Stainless-generated duplicates.
"""

from __future__ import annotations

from datetime import datetime
from typing import NotRequired, Self, TypedDict

from nemo_platform_plugin.entity_naming import NAME_MAX_LENGTH, NAME_PATTERN, NAME_PATTERN_DESCRIPTION
from nemo_platform_plugin.schema import Page
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_serializer, model_validator
from typing_extensions import Required

# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


class PlatformSecretResponse(BaseModel):
    """Response model for a platform secret."""

    name: str = Field(description="The name of the secret")
    workspace: str = Field(description="The workspace ID the secret belongs to")
    description: str | None = Field(default=None, description="An optional description of the secret")
    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)


class PlatformSecretAccessResponse(BaseModel):
    """Response model for accessing a platform secret's value."""

    name: str = Field(description="The name of the secret")
    workspace: str = Field(description="The workspace ID the secret belongs to")
    value: str = Field(description="The payload of the secret")

    @property
    def data(self) -> str:
        return self.value


class PlatformSecretAdminRotationResponse(BaseModel):
    """Response schema for admin secret rotation routine."""

    rotated_secrets: int
    success: bool


class PlatformSecretResponsesPage(Page[PlatformSecretResponse]):
    """Compatibility page model for the legacy SDK secrets resource."""


# ---------------------------------------------------------------------------
# Request types
# ---------------------------------------------------------------------------


class PlatformSecretCreateRequest(BaseModel):
    """Request body for creating a new platform secret."""

    # ``value`` is a ``SecretStr`` so it is masked in reprs/logs, but the
    # ``_serialize_value`` JSON serializer below emits the real plaintext on the
    # wire — without it, ``model_dump_json`` would send ``"**********"`` and the
    # server would store the mask instead of the secret. (Keep this as a comment,
    # not the class docstring, so it does not leak into the OpenAPI schema.)

    model_config = ConfigDict(regex_engine="python-re")

    name: str = Field(
        description=f"The name of the secret to create. {NAME_PATTERN_DESCRIPTION}",
        max_length=NAME_MAX_LENGTH,
        pattern=NAME_PATTERN,
        examples=["hf-token", "wandb-api-key"],
    )
    description: str | None = Field(default=None, description="An optional description of the secret")
    value: SecretStr = Field(description="The payload of the secret")

    @field_serializer("value", when_used="json")
    def _serialize_value(self, value: SecretStr) -> str:
        return value.get_secret_value()

    @model_validator(mode="after")
    def _validate_value(self) -> Self:
        if not self.value.get_secret_value():
            raise ValueError("Secret value cannot be empty")
        return self


class PlatformSecretUpdateRequest(BaseModel):
    """Request body for updating a platform secret's metadata."""

    description: str | None = Field(default=None, description="An optional description of the secret")
    value: SecretStr | None = Field(default=None, description="The new secret value")

    @field_serializer("value", when_used="json")
    def _serialize_value(self, value: SecretStr | None) -> str | None:
        return value.get_secret_value() if value is not None else None

    @model_validator(mode="after")
    def _validate_value(self) -> Self:
        if self.value is not None and not self.value.get_secret_value():
            raise ValueError("Secret value cannot be empty")
        return self


# ---------------------------------------------------------------------------
# Query parameter types
# ---------------------------------------------------------------------------


class ListSecretsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]


class SecretCreateParams(TypedDict, total=False):
    workspace: str
    name: Required[str]
    value: Required[str]
    description: str


class SecretListParams(TypedDict, total=False):
    workspace: str
    page: int
    page_size: int


class SecretUpdateParams(TypedDict, total=False):
    workspace: str
    description: str
    value: str

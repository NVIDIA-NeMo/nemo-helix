# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ResolvedTokenKind = Literal["access_key", "oidc_access_token", "workload_access_token", "workload_subject_token"]


class AuthenticateErrorResponse(BaseModel):
    """Bearer token authentication error response."""

    detail: str


class AuthenticateResponse(BaseModel):
    """Successful bearer token authentication response for direct callers."""

    principal: str
    email: str | None = Field(default=None, json_schema_extra={"nullable": True})
    groups: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    jti: str | None = Field(default=None, json_schema_extra={"nullable": True})
    token_kind: ResolvedTokenKind
    on_behalf_of: str | None = Field(default=None, json_schema_extra={"nullable": True})
    on_behalf_of_email: str | None = Field(default=None, json_schema_extra={"nullable": True})
    on_behalf_of_groups: list[str] = Field(default_factory=list)

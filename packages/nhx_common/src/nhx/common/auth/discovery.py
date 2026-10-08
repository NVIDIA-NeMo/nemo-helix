# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth discovery response models shared by services and clients."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

DEFAULT_AUTH_DISCOVERY_SCOPES = "openid profile email offline_access"
BearerTokenSource = Literal["access_token", "id_token"]


class AuthDiscoveryBearerTokenSourceError(Exception):
    """Raised when auth discovery advertises an unsupported bearer token source."""


def parse_bearer_token_source(value: object) -> BearerTokenSource:
    """Validate a bearer-token response field received from discovery."""
    if value == "access_token":
        return "access_token"
    if value == "id_token":
        return "id_token"
    raise ValueError("OIDC bearer_token_source must be 'access_token' or 'id_token'")


class OIDCDiscoveryResponse(BaseModel):
    """OIDC discovery response returned by the auth discovery endpoint."""

    model_config = ConfigDict(extra="ignore", strict=True)

    issuer: str
    authorization_endpoint: str | None = None
    token_endpoint: str | None = None
    device_authorization_endpoint: str | None = None
    userinfo_endpoint: str | None = None
    client_id: str
    cli_client_id: str | None = None
    bearer_token_source: BearerTokenSource = "access_token"
    device_authorization_requires_device_id: bool = False
    device_authorization_display_name: str | None = None
    device_token_request_includes_scope: bool = True
    default_scopes: str = DEFAULT_AUTH_DISCOVERY_SCOPES
    scope_prefix: str | None = None
    workload_token_exchange_enabled: bool = False
    workload_client_id: str | None = None
    workload_token_endpoint: str | None = None
    workload_audience: str | None = None
    workload_scope: str | None = None

    @field_validator("bearer_token_source", mode="before")
    @classmethod
    def _parse_bearer_token_source(cls, value: object) -> BearerTokenSource:
        try:
            return parse_bearer_token_source(value)
        except ValueError as exc:
            raise AuthDiscoveryBearerTokenSourceError(str(exc)) from exc


class AuthDiscoveryResponse(BaseModel):
    """Auth discovery response returned by the auth discovery endpoint."""

    model_config = ConfigDict(extra="ignore", strict=True)

    auth_enabled: bool
    oidc: OIDCDiscoveryResponse | None = None

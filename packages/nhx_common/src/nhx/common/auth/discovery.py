# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth discovery response models shared by services and clients."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class OidcAdvertisedClientBase(BaseModel):
    """Fields shared by interactive OIDC clients."""

    model_config = ConfigDict(extra="ignore", strict=True)

    client_id: str
    default: bool
    bearer_token_source: BearerTokenSource = "access_token"
    default_scopes: str = DEFAULT_AUTH_DISCOVERY_SCOPES
    scope_prefix: str | None = None

    @field_validator("bearer_token_source", mode="before")
    @classmethod
    def _parse_bearer_token_source(cls, value: object) -> BearerTokenSource:
        try:
            return parse_bearer_token_source(value)
        except ValueError as exc:
            raise AuthDiscoveryBearerTokenSourceError(str(exc)) from exc


class PublicOidcAdvertisedClient(OidcAdvertisedClientBase):
    """Public client using direct provider tokens or NeMo-managed sessions."""

    name: Literal["public"] = "public"
    client_authentication: Literal["public"] = "public"
    server_side_sessions: bool = False
    authorization_endpoint: str | None = None
    token_endpoint: str | None = None
    device_authorization_endpoint: str | None = None
    device_authorization_requires_device_id: bool = False
    device_authorization_display_name: str | None = None
    device_token_request_includes_scope: bool = True
    authorization_start_endpoint: str | None = None
    broker_token_endpoint: str | None = None


class ConfidentialOidcAdvertisedClient(OidcAdvertisedClientBase):
    """Confidential client that talks only to the NeMo Helix broker."""

    name: Literal["confidential"] = "confidential"
    client_authentication: Literal["client_secret_basic"] = "client_secret_basic"
    authorization_start_endpoint: str
    broker_token_endpoint: str


OidcAdvertisedClient = Annotated[
    PublicOidcAdvertisedClient | ConfidentialOidcAdvertisedClient,
    Field(discriminator="name"),
]


class OIDCDiscoveryResponse(BaseModel):
    """OIDC discovery response returned by the auth discovery endpoint."""

    model_config = ConfigDict(extra="ignore", strict=True)

    issuer: str
    userinfo_endpoint: str | None = None
    clients: list[OidcAdvertisedClient] = Field(default_factory=list)
    workload_token_exchange_enabled: bool = False
    workload_client_id: str | None = None
    workload_token_endpoint: str | None = None
    workload_audience: str | None = None
    workload_scope: str | None = None


class AuthDiscoveryResponse(BaseModel):
    """Auth discovery response returned by the auth discovery endpoint."""

    model_config = ConfigDict(extra="ignore", strict=True)

    auth_enabled: bool
    oidc: OIDCDiscoveryResponse | None = None

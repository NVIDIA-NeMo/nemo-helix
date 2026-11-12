# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire types for user profiles (`/me`), preferences, and principal display resolution."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal, TypedDict
from zoneinfo import available_timezones

from pydantic import BaseModel, ConfigDict, Field, JsonValue, RootModel, field_validator

AuthMethod = Literal["web_session", "oidc_bearer", "access_key", "workload_token", "trusted_header", "disabled"]
CallerKind = Literal["principal", "service_principal"]
PreferenceDefaultsSource = Literal["config", "api"]

PREFERENCE_NAMESPACE_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,127}$"
MAX_RESOLVE_PRINCIPALS = 100
# BCP 47 language tag: a 2-3 letter (or 5-8 letter) language plus optional subtags.
_LOCALE_RE = re.compile(r"^([A-Za-z]{2,3}|[A-Za-z]{5,8})([_-][A-Za-z0-9]{2,8})*$")


class Profile(BaseModel):
    """The caller's profile. ``display_name`` is the effective name every client should render."""

    display_name: str = Field(description="Override, else IdP name, else email, else principal id.")
    display_name_override: str | None = Field(default=None, description="Name the user set themselves.")
    idp_display_name: str | None = Field(default=None, description="Name from the identity provider.")
    avatar_url: str | None = Field(default=None, description="Avatar from the identity provider, if any.")
    timezone: str | None = Field(default=None, description="IANA time zone.")
    locale: str | None = Field(default=None, description="BCP 47 language tag.")


class Me(BaseModel):
    """Who the caller is, resolved to a stable account."""

    account_id: str
    principal_id: str = Field(description="Effective principal id (the on-behalf-of subject when delegated).")
    email: str | None = Field(default=None, description="IdP-sourced email. Not editable.")
    groups: list[str] = Field(default_factory=list)
    caller_kind: CallerKind
    auth_method: AuthMethod
    profile: Profile
    can_edit_profile: bool = Field(description="False for delegated callers and service accounts.")
    is_platform_admin: bool


class ProfileUpdate(BaseModel):
    """Fields to change. Omitted fields are unchanged; null clears a field."""

    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, max_length=256, json_schema_extra={"nullable": True})
    timezone: str | None = Field(default=None, max_length=64, json_schema_extra={"nullable": True})
    locale: str | None = Field(default=None, max_length=35, json_schema_extra={"nullable": True})

    @field_validator("display_name")
    @classmethod
    def _strip_display_name(cls, value: str | None) -> str | None:
        return (value or "").strip() or None

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, value: str | None) -> str | None:
        if value is not None and value not in available_timezones():
            raise ValueError("timezone must be an IANA time zone name")
        return value

    @field_validator("locale")
    @classmethod
    def _valid_locale(cls, value: str | None) -> str | None:
        if value is not None and not _LOCALE_RE.match(value):
            raise ValueError("locale must be a BCP 47 language tag")
        return value


class PreferenceValue(RootModel[dict[str, JsonValue]]):
    """A preference document, or an RFC 7396 merge patch for one."""


class PreferenceDocument(BaseModel):
    namespace: str
    workspace: str | None = None
    value: dict[str, Any] = Field(description="The caller's own document; PUT and PATCH act on this.")
    defaults: dict[str, Any] = Field(default_factory=dict, description="Deployment-wide defaults for the namespace.")
    effective: dict[str, Any] = Field(description="defaults merge-patched with value. Render this.")
    db_version: int = Field(description="Version for If-Match; 0 when the caller has no document yet.")
    updated_at: datetime | None = None


class PreferenceDocumentList(BaseModel):
    data: list[PreferenceDocument]


class PreferenceDefaults(BaseModel):
    namespace: str
    value: dict[str, Any]
    managed_by: PreferenceDefaultsSource = Field(
        description="config: set in platform config and read-only through the API; api: set by a HelixAdmin."
    )
    db_version: int = Field(description="Version for If-Match; 0 for config-managed or unset defaults.")


class PreferenceDefaultsList(BaseModel):
    data: list[PreferenceDefaults]


class ProfileResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principals: list[str] = Field(max_length=MAX_RESOLVE_PRINCIPALS)


class ResolvedProfile(BaseModel):
    """Public display fields for one principal string. Unresolved entries have ``display_name: null``."""

    principal: str
    account_id: str | None = None
    display_name: str | None = None
    idp_display_name: str | None = None
    display_name_is_override: bool = False
    avatar_url: str | None = None
    status: str | None = None
    ambiguous: bool = False


class ProfileResolveResponse(BaseModel):
    data: list[ResolvedProfile]


class PreferenceListQueryParams(TypedDict, total=False):
    namespace: str
    workspace: str


class PreferenceQueryParams(TypedDict, total=False):
    workspace: str

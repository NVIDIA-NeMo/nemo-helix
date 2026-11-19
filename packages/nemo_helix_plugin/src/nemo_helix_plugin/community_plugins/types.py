# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request/response types for the external plugin install lifecycle.

Both the platform server (``nhx.platform_runner.community_plugins.installation``)
and the typed client (``community_plugins.client``) import these, so there is one
source of truth for the HTTP contract.
"""

from __future__ import annotations

from nemo_helix_plugin.community_plugins.registry import ExternalPlugin, validate_remote_url
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

TRUST_WARNING = (
    "Installing approves the declared external endpoints and workspace secret use. "
    "A UI bundle runs with Studio-origin privileges inside Studio, not in a sandbox. "
    "A pinned hash verifies identity/integrity, not safety. UI assets are public; never embed secrets."
)


class ManifestSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(max_length=2048)

    @field_validator("url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        return validate_remote_url(value)


class CatalogEntry(ManifestSource):
    name: str = Field(min_length=2, max_length=80)
    title: str = Field(min_length=1, max_length=160)


class CatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entries: tuple[CatalogEntry, ...] = Field(default=(), max_length=256)


class InstallRequest(ManifestSource):
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    consent: bool = Field(strict=True)


class PluginManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = Field(default=1, ge=1, le=1, strict=True)
    plugin: ExternalPlugin


class PreviewResponse(BaseModel):
    manifest: PluginManifest
    manifest_sha256: str
    warning: str = TRUST_WARNING


class InstallResponse(BaseModel):
    name: str
    manifest_sha256: str
    # Serialized as ``bundleUrl``; accept either spelling so the client can parse the wire form.
    bundle_url: str | None = Field(
        validation_alias=AliasChoices("bundle_url", "bundleUrl"), serialization_alias="bundleUrl"
    )


class EnableRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = Field(strict=True)

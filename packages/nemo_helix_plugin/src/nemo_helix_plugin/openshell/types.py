# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared, value-free provider configuration for OpenShell execution backends."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GatewayWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_min_length=1)

    kind: Literal["job", "deployment"]
    workspace: str
    name: str
    id: str
    attempt_id: str | None = None
    step_name: str | None = None


class GatewayProviderBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: str = Field(min_length=1)
    credentials: dict[str, str] = Field(min_length=1)


class GatewayInjectionProfile(BaseModel):
    """Operator approval of a gateway profile for selected Helix workspaces."""

    model_config = ConfigDict(extra="forbid")

    workspaces: set[str] = Field(min_length=1)
    credential_keys: set[str] = Field(min_length=1)
    profile_workspace: str = ""


class ProviderAccessRevoked(RuntimeError):
    """Required providers were removed; existing workload processes may continue."""


class GatewayCredentialConfig(BaseModel):
    """Trusted controller configuration; never copied into a workload."""

    model_config = ConfigDict(extra="forbid")

    gateway_id: str = Field(min_length=1)
    registration_key_file: Path
    profiles: dict[str, GatewayInjectionProfile] = Field(min_length=1)
    renewal_mode: Literal["gateway", "helix"] = Field(
        default="gateway",
        description="Use gateway-managed renewal, or Helix controller updates for active sandboxes, including idle ones.",
    )
    renewal_interval_seconds: int = Field(
        default=10,
        ge=1,
        le=15,
        description="Minimum time between Helix provider updates; allow controller polling and propagation within the configured driver cache lifetime.",
    )
    credential_refresh_verified: bool = Field(
        default=False,
        description="Operator attestation that the selected renewal mode refreshes expiring external credentials and preserves running placeholders on this gateway build.",
    )

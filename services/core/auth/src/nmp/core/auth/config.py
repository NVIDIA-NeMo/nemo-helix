# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Auth service (v2)."""

from typing import Literal, Optional

from nmp.common.config import AuthConfig as SharedAuthConfig
from pydantic import Field


class AuthServiceConfig(SharedAuthConfig):
    """
    Configuration for the Auth Service.

    Extends the shared AuthConfig (which has enabled, policy_decision_point_base_url,
    policy_decision_point_provider) with auth-service-specific fields.

    Environment variables use the NMP_AUTH_ prefix.
    """

    port: int = Field(
        default=8000,
        description="Port to run the service on",
    )

    # Refresh interval for policy data (seconds) - only used when provider=embedded
    policy_data_refresh_interval: float = Field(
        default=30,
        description="Refresh interval for policy data in seconds",
    )

    bundle_cache_seconds: float = Field(
        default=5,
        description="Seconds to cache the OPA bundle",
    )

    # Platform admin bootstrap
    admin_email: Optional[str] = Field(
        default=None,
        description="Bootstrap admin email for platform setup",
    )

    # Default workspace for all-users Editor role
    default_workspace: str = Field(
        default="default",
        description="Name of the default workspace where all authenticated users get Editor role",
    )

    # Embedded PDP resource limits
    embedded_pdp_cpu_limit: int = Field(
        default=200,
        description="CPU budget for embedded PDP policy evaluation, in millions of WASM fuel units. "
        "Default of 200 provides headroom for full plugin-merged authorization data.",
    )
    embedded_pdp_memory_limit_mb: int = Field(
        default=32,
        description="Maximum linear memory (MB) the embedded PDP WASM runtime can consume.",
    )

    # Plugin HTTP authz fail-mode: what to do when a plugin contributes invalid authz
    # (an unruled route, or a rule referencing an undeclared / out-of-namespace permission).
    # The offending routes are always emitted as explicit denies; this controls the blast
    # radius. hard_fail: refuse to build the OPA bundle (default — fail closed at the platform
    # level, matching the 743 spec's "a missing path rule is a validation error"). quarantine:
    # deny the whole offending plugin but keep the platform up. deny_route: deny only the bad
    # routes. A deployment that loads dynamically-discovered or third-party plugins CI never
    # vetted can downgrade to quarantine/deny_route so one bad plugin can't wedge the platform.
    on_invalid_plugin: Literal["deny_route", "quarantine", "hard_fail"] = Field(
        default="hard_fail",
        description="Fail-mode for a plugin that contributes invalid HTTP authz.",
    )

    allowed_service_principals: list[str] = Field(
        default_factory=list,
        description=(
            "Additional service principal names allowed for stable account materialization. "
            "Built-in platform services and installed plugin services are allowed automatically."
        ),
    )


# Backward compatibility alias
AuthConfig = AuthServiceConfig

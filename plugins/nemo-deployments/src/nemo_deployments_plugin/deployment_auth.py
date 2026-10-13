# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared deployment auth planning helpers."""

from __future__ import annotations

from dataclasses import dataclass

from nemo_deployments_plugin.auth_proxy import auth_proxy_port
from nemo_helix_plugin.auth import AuthContext, platform_auth_enabled
from nemo_helix_plugin.auth.workload_identity import is_workload_identity_token_exchange_enabled


@dataclass(frozen=True)
class DeploymentAuthPlan:
    """Resolved platform-auth wiring for a deployment workload."""

    auth_proxy_base_url: str | None = None
    auth_proxy_identity: str | None = None
    auth_proxy_on_behalf_of: str | None = None
    workload_identity_enabled: bool = False
    error: str | None = None
    warning: str | None = None


def plan_deployment_auth(
    *,
    service_identity: str,
    on_behalf_of: str | None,
    auth_context: AuthContext | None,
    workload_name: str,
    workload_label: str,
) -> DeploymentAuthPlan:
    """Return the auth-proxy/workload-identity plan for a deployment workload.

    Callers choose the workload-specific service identity and fallback platform
    URL. This helper owns the shared relationship between legacy auth-proxy
    on-behalf-of delegation and workload identity token exchange, so workload
    backends do not need to branch on platform auth modes directly.
    """
    token_exchange_enabled = is_workload_identity_token_exchange_enabled()
    workload_identity_enabled = token_exchange_enabled and auth_context is not None

    if not platform_auth_enabled():
        return DeploymentAuthPlan(workload_identity_enabled=workload_identity_enabled)

    if token_exchange_enabled and auth_context is None:
        return DeploymentAuthPlan(
            error=(
                f"{workload_label} requires creator auth context when workload token exchange is enabled; "
                "refusing to deploy without creator workload identity context."
            )
        )

    auth_proxy_on_behalf_of = None if token_exchange_enabled else on_behalf_of
    warning = None
    if not token_exchange_enabled and auth_proxy_on_behalf_of is None:
        warning = (
            f"{workload_label} {workload_name!r} has no creator principal; the workload will run as the "
            f"unscoped {service_identity} service principal without on-behalf-of delegation."
        )

    return DeploymentAuthPlan(
        auth_proxy_base_url=f"http://127.0.0.1:{auth_proxy_port()}",
        auth_proxy_identity=service_identity,
        auth_proxy_on_behalf_of=auth_proxy_on_behalf_of,
        workload_identity_enabled=workload_identity_enabled,
        warning=warning,
    )

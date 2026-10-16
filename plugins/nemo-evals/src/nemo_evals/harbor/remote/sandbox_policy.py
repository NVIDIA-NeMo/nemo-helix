# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze an operator sandbox profile into a run's plan (M2 story 3's resolver, trimmed for M2 story 0).

The MVP picks a profile by the request's ``execution.sandbox_policy`` name and applies no request
overrides or ceilings; M2 story 3 adds both without changing the frozen shape.
"""

from nemo_evals.config import HarborTaskJobsConfig
from nemo_evals.harbor.remote.profile import OpenSandboxConnection, ResolvedHarborSandboxPolicy
from nemo_helix_plugin.config import NemoHelixConfig


class SandboxPolicyError(ValueError):
    """The request names a profile the deployment can't run. Submit reports it as a 422."""


def resolve_sandbox_policy(
    harbor: HarborTaskJobsConfig, platform: NemoHelixConfig, profile_name: str
) -> ResolvedHarborSandboxPolicy:
    """Copy profile ``profile_name`` and the platform's OpenSandbox server into a frozen policy.

    Raises:
        SandboxPolicyError: The profile is unknown, or the platform has no OpenSandbox server.
    """
    profile = harbor.sandbox_profiles.get(profile_name)
    if profile is None:
        names = ", ".join(sorted(harbor.sandbox_profiles))
        raise SandboxPolicyError(f"unknown sandbox_policy {profile_name!r}; this deployment offers: {names}")
    if not platform.sandbox_cluster_capable or not platform.sandbox_server_domain:
        raise SandboxPolicyError("OpenSandbox isn't configured on this platform")
    protocol = platform.sandbox_server_protocol or "http"
    if protocol not in ("http", "https"):
        raise SandboxPolicyError(f"platform sandbox_server_protocol {protocol!r} must be http or https")
    connection = OpenSandboxConnection(
        **profile.connection.model_dump(),
        domain=platform.sandbox_server_domain,
        protocol=protocol,
    )
    return ResolvedHarborSandboxPolicy(
        **profile.model_dump(exclude={"connection"}),
        profile=profile_name,
        connection=connection,
    )

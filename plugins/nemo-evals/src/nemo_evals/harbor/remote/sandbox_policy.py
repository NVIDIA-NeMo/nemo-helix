# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze an operator sandbox profile into a run's plan (M2 story 3's resolver, trimmed for M2 story 0).

The MVP picks a profile by the request's ``execution.sandbox.profile`` name, or the operator's
``default_sandbox_profile`` when it names none, and freezes the request's ``execution.sandbox`` resource
overrides once they pass the profile's ceilings. Egress overrides and
per-task resource checks against task.toml come with M2 story 3 and M3.
"""

from nemo_evals.config import HarborSandboxProfileConfig, HarborTaskJobsConfig
from nemo_evals.harbor.remote.profile import (
    OpenSandboxConnection,
    ResolvedHarborSandboxPolicy,
    ResourceOverrides,
    SandboxSelection,
)
from nemo_helix_plugin.config import NemoHelixConfig


class SandboxPolicyError(ValueError):
    """The request names a profile the deployment can't run. Submit reports it as a 422."""


def resolve_sandbox_policy(
    harbor: HarborTaskJobsConfig,
    platform: NemoHelixConfig,
    selection: SandboxSelection | None = None,
) -> ResolvedHarborSandboxPolicy:
    """Copy the selected profile, the request's resource overrides and the platform's OpenSandbox server into a frozen policy.

    Raises:
        SandboxPolicyError: The profile is unknown, an override exceeds the profile's ceilings, or the
            platform has no OpenSandbox server.
    """
    selection = selection or SandboxSelection()
    profile_name = selection.profile or harbor.default_sandbox_profile
    profile = harbor.sandbox_profiles.get(profile_name)
    if profile is None:
        names = ", ".join(sorted(harbor.sandbox_profiles))
        raise SandboxPolicyError(f"unknown execution.sandbox.profile {profile_name!r}; this deployment offers: {names}")
    overrides = ResourceOverrides.model_validate(selection.model_dump(exclude={"profile"}))
    _check_overrides(overrides, profile, profile_name)
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
        resource_overrides=overrides,
    )


def _check_overrides(overrides: ResourceOverrides, profile: HarborSandboxProfileConfig, profile_name: str) -> None:
    """Reject any override above its ceiling; the first violation in field order is reported.

    A storage override is also capped by an enforced ``max_disk_mb``, so the sandbox never gets less
    disk than the run asked for.
    """
    storage_ceilings = [("max_storage_mb", profile.max_storage_mb)]
    if profile.enforce_disk_limit:
        storage_ceilings.append(("max_disk_mb", profile.max_disk_mb))
    checks = [
        ("cpus", overrides.cpus, [("max_cpus", profile.max_cpus)]),
        ("memory_mb", overrides.memory_mb, [("max_memory_mb", profile.max_memory_mb)]),
        ("storage_mb", overrides.storage_mb, storage_ceilings),
    ]
    for field, value, ceilings in checks:
        limits = [(name, limit) for name, limit in ceilings if limit is not None]
        if value is None or not limits:
            continue
        name, limit = min(limits, key=lambda item: item[1])
        if value > limit:
            raise SandboxPolicyError(
                f"execution.sandbox.{field}={value} exceeds sandbox profile {profile_name!r}'s {name}={limit}"
            )

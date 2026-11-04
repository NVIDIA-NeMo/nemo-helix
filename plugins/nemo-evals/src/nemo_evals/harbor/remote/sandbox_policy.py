# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze an operator sandbox profile into a run's plan, and check each task's sandboxes against its ceilings.

The MVP picks a profile by the request's ``execution.sandbox.profile`` name, or the operator's
``default_sandbox_profile`` when it names none, and freezes the request's ``execution.sandbox`` resource
overrides once they pass the profile's ceilings. ``task_resources_violation`` then checks every task:
its agent sandbox (overrides, else task.toml, else the defaults) and a separate verifier (task.toml, else
the defaults).

- ``max_cpus``, ``max_memory_mb`` and ``max_storage_mb`` bound the requested values; storage is checked
  before any disk clamp.
- ``max_disk_mb`` (with ``enforce_disk_limit``) clamps task.toml's storage, and rejects only an override
  above it, so the sandbox never gets less disk than the run asked for.
- ``max_gpus`` and ``gpu_types`` bound each sandbox's task.toml ``gpus`` and the GPU types it accepts.
"""

from decimal import Decimal
from typing import Literal

from harbor.models.task.config import TaskConfig
from nemo_evals.config import HarborSandboxProfileConfig, HarborTaskJobsConfig
from nemo_evals.harbor.remote.profile import (
    OpenSandboxConnection,
    ResolvedHarborSandboxPolicy,
    ResourceOverrides,
    SandboxSelection,
)
from nemo_evals.harbor.remote.resources import task_sandboxes
from nemo_helix_plugin.config import NemoHelixConfig
from nhx_sandbox.harbor_opensandbox_services import ComposeServices, quantity_value


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


def task_resources_violation(task_config: TaskConfig, policy: ResolvedHarborSandboxPolicy) -> str | None:
    """Why one of the task's sandboxes would exceed the profile's ceilings, or None."""
    ceilings = {"cpus": policy.max_cpus, "memory_mb": policy.max_memory_mb, "storage_mb": policy.max_storage_mb}
    for sandbox in task_sandboxes(task_config, policy):
        for field, source in sandbox.sources.items():
            value, limit = getattr(sandbox.config, field), ceilings[field]
            if limit is not None and value > limit:
                return (
                    f"{sandbox.role} sandbox {field}={value} ({source}) exceeds sandbox profile "
                    f"{policy.profile!r}'s max_{field}={limit}"
                )
    return None


GpuRefusal = Literal["gpu_unsupported", "gpu_request_exceeds_policy", "unsupported_gpu_type"]


def task_gpus_violation(task_config: TaskConfig, policy: ResolvedHarborSandboxPolicy) -> tuple[GpuRefusal, str] | None:
    """Why the profile can't give one of the task's sandboxes the GPUs it asks for, as a refusal code and detail.

    A separate verifier asks for its own GPUs. A task's ``gpu_types`` lists the types it accepts, so one that
    names none of the profile's types is refused rather than run on another GPU; one that names none accepts any.
    """
    profile_types = {gpu_type.casefold() for gpu_type in policy.gpu_types}
    for sandbox in task_sandboxes(task_config, policy):
        gpus = sandbox.config.gpus or 0
        if gpus == 0:
            continue

        if policy.max_gpus == 0:
            return "gpu_unsupported", f"sandbox profile {policy.profile!r} has no GPUs (max_gpus=0)"

        if gpus > policy.max_gpus:
            return (
                "gpu_request_exceeds_policy",
                f"{sandbox.role} sandbox gpus={gpus} exceeds sandbox profile {policy.profile!r}'s "
                f"max_gpus={policy.max_gpus}",
            )

        accepted = sandbox.config.gpu_types or []
        if accepted and not profile_types & {gpu_type.casefold() for gpu_type in accepted}:
            offered = ", ".join(policy.gpu_types) or "none listed"
            return (
                "unsupported_gpu_type",
                f"{sandbox.role} sandbox accepts GPU types {', '.join(accepted)}; sandbox profile "
                f"{policy.profile!r} has {offered}",
            )

    return None


def compose_resources_violation(services: ComposeServices, policy: ResolvedHarborSandboxPolicy) -> str | None:
    """Why a compose service's declared limits, requests or ``shm_size`` exceed the profile's ceilings, or None.

    Only what the task declares is checked, one service at a time: a service that sets nothing runs without
    limits, as it does today. Memory is compared in MiB. ``shm_size`` is memory-backed, so it counts as memory.
    """
    # Each Kubernetes resource, the profile field that bounds it, that bound, and the size of the bound's unit.
    ceilings = {
        "cpu": ("cpus", policy.max_cpus, Decimal(1)),
        "memory": ("memory_mb", policy.max_memory_mb, Decimal(2**20)),
    }

    for service in services.services:
        if service.shm_size is not None and policy.max_memory_mb is not None:
            if quantity_value(service.shm_size) / Decimal(2**20) > policy.max_memory_mb:
                return (
                    f"compose service {service.name} shm_size={service.shm_size} exceeds sandbox profile "
                    f"{policy.profile!r}'s max_memory_mb={policy.max_memory_mb}"
                )

        if service.resources is None:
            continue

        declared = {"limit": service.resources.limits or {}, "request": service.resources.requests or {}}
        for kind, quantities in declared.items():
            for resource, (field_name, ceiling, unit_size) in ceilings.items():
                quantity = quantities.get(resource)
                if ceiling is None or quantity is None:
                    continue

                if quantity_value(quantity) / unit_size > ceiling:
                    return (
                        f"compose service {service.name} {resource} {kind}={quantity} exceeds sandbox profile "
                        f"{policy.profile!r}'s max_{field_name}={ceiling}"
                    )
    return None

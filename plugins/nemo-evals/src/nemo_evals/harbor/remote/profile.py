# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen OpenSandbox policy a taskset run carries in its plan.

Contract module (M1 story 5). The operator config that produces the policy lives in
``nemo_evals.config``; ``resolve_sandbox_policy`` is M2 story 0's slice of M2 story 3's
resolver: it picks a profile by name (the operator's default when the request names none) and
applies the request's resource overrides within the profile's ceilings.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ResourceClass = Literal["cpu", "gpu"]

PINNED_IMAGE_REF_PATTERN = r"@sha256:[0-9a-f]{64}$"
"""An image reference pinned by digest: ``registry/repo@sha256:<64 lowercase hex>``."""


class OpenSandboxClientSettings(BaseModel):
    """Client tuning per sandbox profile (operator config, M2 story 3). No server address, no key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_timeout_s: float = Field(
        default=960.0, gt=0, description="Must exceed the server's sandbox_create_timeout_seconds."
    )
    use_server_proxy: bool = Field(
        default=True, description="Reach execd through the server; Jobs pods can't reach sandbox pod IPs."
    )


class OpenSandboxConnection(OpenSandboxClientSettings):
    """Where dispatchers reach the OpenSandbox server.

    Domain and protocol are copied from the platform's ``sandbox_*`` config at submit; Jobs injects
    the API key into every Job pod, so the plan never names it.
    """

    domain: str = Field(
        min_length=1,
        description=(
            "Platform `sandbox_server_domain` at submit: in-cluster Service DNS, no scheme. The dispatcher "
            "fails if the live injected OPEN_SANDBOX_DOMAIN differs."
        ),
    )
    protocol: Literal["http", "https"] = "http"


class ResourceOverrides(BaseModel):
    """Request-level resource overrides; beat task.toml ``[environment]`` in the agent sandbox only.

    A separate verifier keeps task.toml's resources. Field names match task.toml (M2 story 3).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cpus: int | None = Field(default=None, gt=0)
    memory_mb: int | None = Field(default=None, gt=0)
    storage_mb: int | None = Field(default=None, gt=0)


class SandboxSelection(ResourceOverrides):
    """``execution.sandbox``: an operator sandbox profile by name, and resource overrides within its ceilings."""

    profile: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Name of an operator sandbox profile: the sandbox's isolation and egress rules and its resource "
            "ceilings. Omit for the operator's default profile."
        ),
        examples=["harbor-standard"],
    )


class ResolvedHarborSandboxPolicy(BaseModel):
    """The sandbox profile submit chose, copied into the plan so config edits never reach a running run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: str = Field(min_length=1, description="Profile name, for display and provenance.")
    connection: OpenSandboxConnection
    egress_allow: tuple[str, ...] = Field(description="Operator (Helm) hosts: always applied.")
    request_egress_allow: tuple[str, ...] = Field(
        default=(), description="execution.sandbox.egress_allow, kept apart for provenance."
    )
    agent_channel: tuple[str, ...] = Field(
        default=(), description="Hosts a no-network task's sandbox may still reach (operator config)."
    )
    agent_setup_egress: tuple[str, ...] = Field(
        default=(), description="Hosts the sandbox may also reach during the agent's setup() only (operator config)."
    )
    enforce_closed_book: bool = Field(
        default=False,
        description="Refuse no-network tasks at submit without a closed-book model route (operator config).",
    )
    override_task_egress: bool = False
    allow_task_egress_extension: bool = False
    resource_overrides: ResourceOverrides = ResourceOverrides()
    max_cpus: int | None = None
    max_memory_mb: int | None = None
    max_storage_mb: int | None = None
    enforce_disk_limit: bool = Field(description="Cap each sandbox's ephemeral storage at its storage_mb.")
    max_disk_mb: int | None = Field(description="Clamp on that disk limit; None applies storage_mb as given.")
    egress_verification: Literal["default_action", "strict"]
    max_sandbox_lifetime_s: int = Field(ge=1)
    scoring_reserve_s: int = Field(ge=0)
    unschedulable_after_s: int = Field(ge=1)
    stale_attempt_cleanup_timeout_s: int = Field(ge=1)
    image_reference_mode: Literal["digest", "tag"] = "digest"


DEFAULT_SANDBOX_CPUS = 1
DEFAULT_SANDBOX_MEMORY_MB = 2048
"""What a sandbox gets when neither task.toml nor the request sets cpus or memory_mb: the OpenSandbox SDK's
own fallback for a create with no resources. Filled in explicitly, because the SDK applies it only when
no resource at all is set, and a disk limit alone would leave the sandbox without CPU or memory limits."""


def resource_defaults(policy: ResolvedHarborSandboxPolicy) -> dict[str, int]:
    """The ``cpus`` and ``memory_mb`` a sandbox gets when nothing sets them, capped by the profile's ceilings."""
    cpus, memory_mb = DEFAULT_SANDBOX_CPUS, DEFAULT_SANDBOX_MEMORY_MB
    if policy.max_cpus is not None:
        cpus = min(cpus, policy.max_cpus)
    if policy.max_memory_mb is not None:
        memory_mb = min(memory_mb, policy.max_memory_mb)
    return {"cpus": cpus, "memory_mb": memory_mb}

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen OpenSandbox policy a taskset run carries in its plan.

Contract module (M1 story 5). The operator config that produces the policy lives in
``nemo_evals.config``; ``resolve_sandbox_policy`` picks a profile by name.

TODO(M2 story 3): request resource overrides and the profile's ceilings.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    """Request-level resource overrides; beat task.toml ``[environment]``. Field names match task.toml (M2 story 3)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cpus: int | None = Field(default=None, gt=0)
    memory_mb: int | None = Field(default=None, gt=0)
    storage_mb: int | None = Field(default=None, gt=0)
    gpus: int | None = Field(default=None, ge=0)

    def harbor_kwargs(self) -> dict[str, int]:
        """Harbor's environment kwargs: ``cpus`` → ``override_cpus``, unset fields omitted."""
        return {f"override_{k}": v for k, v in self.model_dump(exclude_none=True).items()}


class ResourceClassPolicy(BaseModel):
    """Sandbox template and task-Job profile for one resource class."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    template: str = Field(min_length=1, description="OpenSandbox BatchSandbox template.")
    dispatcher_profile: str = Field(min_length=1, description="Jobs execution profile for this class's task Jobs.")
    gpu_type_labels: dict[str, list[str]] = Field(default_factory=dict)
    max_gpus_per_task: int = Field(default=0, ge=0)


class ResolvedHarborSandboxPolicy(BaseModel):
    """The sandbox profile submit chose, copied into the plan so config edits never reach a running run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: str = Field(min_length=1, description="Profile name, for display and provenance.")
    connection: OpenSandboxConnection
    resource_classes: dict[ResourceClass, ResourceClassPolicy]
    root_template: str | None
    egress_allow: tuple[str, ...] = Field(description="Operator (Helm) hosts: always applied.")
    request_egress_allow: tuple[str, ...] = Field(
        default=(), description="execution.sandbox.egress_allow, kept apart for provenance."
    )
    agent_channel: tuple[str, ...] = Field(
        default=(), description="Hosts a no-network task's sandbox may still reach (operator config)."
    )
    override_task_egress: bool = False
    allow_task_egress_extension: bool = False
    resource_overrides: ResourceOverrides = ResourceOverrides()
    max_cpus: int | None = None
    max_memory_mb: int | None = None
    max_storage_mb: int | None = None
    egress_verification: Literal["default_action", "strict"]
    max_sandbox_lifetime_s: int = Field(ge=1)
    scoring_reserve_s: int = Field(ge=0)
    unschedulable_after_s: int = Field(ge=1)
    stale_attempt_cleanup_timeout_s: int = Field(ge=1)
    image_reference_mode: Literal["digest", "tag"] = "digest"

    @model_validator(mode="after")
    def _cpu_class_required(self) -> Self:
        if "cpu" not in self.resource_classes:
            raise ValueError("resource_classes must include 'cpu'")
        return self

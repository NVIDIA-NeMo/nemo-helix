# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration namespace for the evals plugin."""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Self

from nemo_evals.harbor.remote.profile import (
    PINNED_IMAGE_REF_PATTERN,
    OpenSandboxClientSettings,
    ResourceClass,
    ResourceClassPolicy,
)
from nemo_helix_plugin.config import NemoConfig, get_platform_config
from pydantic import BaseModel, ConfigDict, Field, model_validator


class HarborSandboxProfileConfig(BaseModel):
    """One named OpenSandbox profile for Harbor task Jobs (M2 story 3, trimmed for M2 story 0).

    The server address and API key come from the platform's ``sandbox_*`` settings, not from here:
    a profile only tunes the client and sets templates, egress and lifetimes.
    """

    model_config = ConfigDict(extra="forbid")

    connection: OpenSandboxClientSettings = Field(default_factory=OpenSandboxClientSettings)
    resource_classes: dict[ResourceClass, ResourceClassPolicy] = Field(
        description="Sandbox template and task-Job profile per resource class; `cpu` is required."
    )
    root_template: str | None = Field(default=None, description="Template for images that must run as root.")
    egress_allow: tuple[str, ...] = Field(
        default=(),
        description="Hostnames, IPs or CIDRs every sandbox may reach; no port. Everything else is denied.",
    )
    agent_channel: tuple[str, ...] = Field(
        default=(),
        description=(
            "Hosts a no-network task's sandbox may still reach (e.g. the inference gateway), so an agent "
            "running in the sandbox can call its model. Empty keeps no-network tasks fully closed."
        ),
    )
    egress_verification: Literal["default_action", "strict"] = "strict"
    max_sandbox_lifetime_s: int = Field(
        default=8 * 3600 + 30 * 60, ge=1, description="OpenSandbox TTL backstop: longest trial plus margin."
    )
    scoring_reserve_s: int = Field(default=600, ge=0)
    unschedulable_after_s: int = Field(default=600, ge=1)
    stale_attempt_cleanup_timeout_s: int = Field(default=300, ge=1)
    image_reference_mode: Literal["digest", "tag"] = "digest"

    @model_validator(mode="after")
    def _cpu_class_required(self) -> Self:
        if "cpu" not in self.resource_classes:
            raise ValueError("resource_classes must include 'cpu'")
        return self


class HarborTaskJobsConfig(BaseModel):
    """Operator settings that enable Kubernetes taskset runs (``evals.runs.harbor``)."""

    model_config = ConfigDict(extra="forbid")

    dispatcher_image: str = Field(
        pattern=PINNED_IMAGE_REF_PATTERN,
        description="Task Job image, pinned by digest: nemo-evals plus the Harbor OpenSandbox environment.",
    )
    dispatcher_profiles: tuple[str, ...] = Field(
        min_length=1, description="Jobs execution profiles a request may name as execution.dispatcher_profile."
    )
    sandbox_profiles: dict[str, HarborSandboxProfileConfig] = Field(
        min_length=1, description="Named sandbox profiles; a request picks one with execution.sandbox_policy."
    )
    allowed_registries: list[str] = Field(
        default_factory=list,
        description="Registries a task's prebuilt docker_image may come from. Empty allows none.",
    )


class TasksetRunsConfig(BaseModel):
    """Operator budgets under evals.runs.

    Construction produces validated execution and finalization budgets for submit;
    the default execution timeout cannot exceed the maximum callers may request.

    Environment overrides use JSON, e.g.
    ``NEMO_EVALS_RUNS='{"default_timeout_seconds": 7}'``: the settings delimiter
    splits underscores, so flat environment names cannot address these fields.
    """

    model_config = ConfigDict(extra="forbid")

    default_timeout_seconds: int = Field(
        default=86_400,
        gt=0,
        description="Run budget when the request omits timeout_seconds (24 h; TB3 with 10 slots needs ~16 h).",
    )
    max_timeout_seconds: int = Field(
        default=259_200, gt=0, description="Largest timeout_seconds a request may ask for (72 h)."
    )
    finalization_grace_seconds: int = Field(
        default=600,
        gt=0,
        description="Time after the execution deadline to finish scoring and bundling (10 min).",
    )
    harbor: HarborTaskJobsConfig | None = Field(
        default=None, description="Kubernetes taskset runs; None leaves them off in this deployment."
    )

    @model_validator(mode="after")
    def _default_within_max(self) -> Self:
        """Validate the relationship between this instance's parsed execution budgets.

        Returns:
            These settings unchanged when the default timeout fits within the maximum.

        Raises:
            ValueError: The default timeout exceeds the maximum allowed timeout.
        """
        if self.default_timeout_seconds > self.max_timeout_seconds:
            raise ValueError("runs.default_timeout_seconds must be ≤ runs.max_timeout_seconds")
        return self


class EvaluatorConfig(NemoConfig):
    """Configuration namespace for the evals plugin."""

    plugin_name: ClassVar[str] = "evals"
    plugin_description: ClassVar[str] = "Configuration namespace for the evals plugin."

    runs: TasksetRunsConfig = Field(
        default_factory=TasksetRunsConfig,
        description="Operator execution timeout defaults, maximum request budget and finalization grace for taskset runs.",
    )

    gym_tasks_image: str | None = Field(
        default=None,
        description=(
            "Optional fully qualified image reference for colocated Gym agent-evaluation jobs. Sandboxed Gym "
            "jobs use the platform CPU tasks image and launch Gym in `sandbox_runtime_image`. Override with "
            "NEMO_EVALS_GYM_TASKS_IMAGE; when set, this bypasses platform image registry/tag qualification."
        ),
    )
    # --- Sandboxed Gym execution -------------------------------------------------------------
    #
    # Whether a Gym evaluation runs its environment inside a sandbox is a property of the
    # deployment, not of the job: the same submitted `GymRunnerTarget` runs colocated on a trusted
    # dev box and sandboxed on a shared cluster. Customizer settled the same question the same way
    # for GRPO (`NHX_RL_SANDBOXED_GYM_DEFAULT`), so the submit contract stays identical across both.

    sandboxed_gym_default: bool = Field(
        default=False,
        description="Run Gym evaluations inside a sandboxed Gym host instead of the job container. "
        "Defaults off, unlike the RL equivalent: an existing evaluator deployment has no OpenSandbox "
        "to provision against, so defaulting on would break every one of them.",
    )
    sandbox_cluster_capable: bool = Field(
        default=False,
        description="Assert that this cluster can actually provision sandboxes. A shared cluster fails "
        "closed until an operator sets this, so a misconfigured deployment refuses the job rather than "
        "silently running user environment code beside the job's own credentials.",
    )
    sandbox_job_storage_pvc_claim: str | None = Field(
        default=None,
        description="PVC claim the sandboxed Gym host mounts for the environment and dataset.",
    )
    sandbox_runtime_image: str | None = Field(
        default=None,
        description="Optional fully qualified image for the sandboxed Gym host. When unset, the deployment "
        "uses the qualified `nhx-gym-host` image matching the platform registry and tag. Set this only to "
        "override that release-coupled default.",
    )
    sandbox_episode_backend: Literal["opensandbox", "memory"] = Field(
        default="opensandbox",
        description="Backend the episode broker provisions nested episode sandboxes through. `memory` "
        "provisions nothing and applies no isolation; it exists so a local deployment can exercise the "
        "path without a cluster, and requires the explicit opt-in below.",
    )
    sandbox_allow_insecure_memory_backend: bool = Field(
        default=False,
        description="Second key required to select the `memory` episode backend. Two keys rather than one "
        "because a single mistyped setting must not silently disable episode isolation.",
    )
    sandbox_host_provider: str = Field(
        default="opensandbox",
        description="Which job-host provider provisions the sandboxed Gym host. `docker` runs it as a "
        "local container for development: it executes the same runtime image contract but is not an "
        "isolation boundary and enforces no egress policy, so it is not a shared-cluster option.",
    )
    sandbox_host_provider_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific settings, e.g. {'root_dir': '/tmp/nhx-gym-host'} for `docker`.",
    )
    sandbox_environment_sub_path: str = Field(
        default="environment",
        description="Sub-path within the job-storage PVC holding the Gym environment, mounted read-only. "
        "Distinct from the workspace sub-path by default: the two mounts share one claim, so equal "
        "sub-paths would mount the same directory twice and let the writable mount modify the "
        "environment the run is supposed to be evaluating.",
    )
    sandbox_workspace_sub_path: str = Field(
        default="workspace",
        description="Sub-path within the job-storage PVC the Gym host writes to, mounted read-write.",
    )
    sandbox_resources: dict[str, str] | None = Field(
        default=None,
        description="Resource requests for the sandboxed Gym host, e.g. {'cpu': '2', 'memory': '8Gi'}. "
        "Left unset the host is scheduled with no request, which on a shared cluster means it competes "
        "unbounded and is first to be evicted. No default is guessed: a Gym host's footprint depends on "
        "the environment it runs.",
    )
    sandbox_policy_base_urls: tuple[str, ...] = Field(
        default=(),
        description="Model/inference endpoints the sandboxed Gym host may reach, as URLs. Each becomes an "
        "egress allowance for its host and port. Deployment config rather than a job field on purpose: a "
        "job that could widen its own sandbox's egress would defeat the isolation it is running under.",
    )
    sandbox_egress_allow: tuple[str, ...] = Field(
        default=(),
        description="Additional egress destinations as `host:port`, for anything the environment needs "
        "that is not a model endpoint. The episode broker's own address is always allowed and does not "
        "need listing.",
    )
    sandbox_approved_images: tuple[str, ...] = Field(
        default=(),
        description="Images the episode broker will provision for nested episode sandboxes. Empty means "
        "every episode create is refused, matching the broker's own closed default -- a run that needs "
        "episodes must be granted its images explicitly.",
    )
    allow_insecure_cloudpickle_metrics: bool = Field(
        default=False,
        description="Accept and run cloudpickle metric bundles. A cloudpickle bundle is arbitrary Python "
        "supplied by the caller, and it runs with the Evals service's credentials, which reach every "
        "workspace. Enable only where every principal allowed to create metrics or submit Evals jobs "
        "is trusted to run code as the service. Job workers do not read the platform config file: set "
        "`NEMO_EVALS_ALLOW_INSECURE_CLOUDPICKLE_METRICS` in each job backend's "
        "`jobs.executor_defaults.<backend>.env` as well. This setting is the access control: submitters "
        "choose their job's executor profile, so opting in only some profiles restricts nothing.",
    )


def get_config() -> EvaluatorConfig:
    """Return the Evals plugin configuration singleton."""
    return EvaluatorConfig.get()


config = get_config()
platform_config = get_platform_config()

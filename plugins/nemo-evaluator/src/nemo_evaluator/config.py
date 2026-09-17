# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration namespace for the evaluator plugin."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from nemo_platform_plugin.config import NemoConfig, get_platform_config
from pydantic import Field


class EvaluatorConfig(NemoConfig):
    """Configuration namespace for the evaluator plugin."""

    plugin_name: ClassVar[str] = "evaluator"
    plugin_description: ClassVar[str] = "Configuration namespace for the evaluator plugin."

    gym_tasks_image: str | None = Field(
        default=None,
        description=(
            "Optional fully qualified image reference for colocated Gym agent-evaluation jobs. Sandboxed Gym "
            "jobs use the platform CPU tasks image and launch Gym in `sandbox_runtime_image`. Override with "
            "NEMO_EVALUATOR_GYM_TASKS_IMAGE; when set, this bypasses platform image registry/tag qualification."
        ),
    )
    # --- Sandboxed Gym execution -------------------------------------------------------------
    #
    # Whether a Gym evaluation runs its environment inside a sandbox is a property of the
    # deployment, not of the job: the same submitted `GymRunnerTarget` runs colocated on a trusted
    # dev box and sandboxed on a shared cluster. Customizer settled the same question the same way
    # for GRPO (`NMP_RL_SANDBOXED_GYM_DEFAULT`), so the submit contract stays identical across both.

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
        "uses the qualified `nmp-gym-host` image matching the platform registry and tag. Set this only to "
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
        description="Provider-specific settings, e.g. {'root_dir': '/tmp/nmp-gym-host'} for `docker`.",
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


def get_config() -> EvaluatorConfig:
    """Return the Evaluator plugin configuration singleton."""
    return EvaluatorConfig.get()


config = get_config()
platform_config = get_platform_config()

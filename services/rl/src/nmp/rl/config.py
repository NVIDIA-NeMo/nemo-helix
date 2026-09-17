# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the nmp-rl compiler and container tasks.

Modeled after :mod:`nmp.unsloth.config`. Environment variables use the
``NMP_RL_`` prefix and drive image resolution and the multi-node shared-storage
gate for the ``PlatformJobSpec`` the plugin's ``RlJob.compile`` builds.
"""

from nmp.common.config import create_service_config_class, get_platform_config, get_service_config
from pydantic import Field


class RlConfig(create_service_config_class("rl")):  # type: ignore[misc]  # ty: ignore[unsupported-base]
    """Environment variables use the ``NMP_RL_`` prefix."""

    image_registry: str | None = Field(
        default=None,
        description=(
            "Registry host/path prefix for nmp-customizer-tasks and nmp-rl-training. "
            "Override via NMP_RL_IMAGE_REGISTRY; defaults to the platform's image registry."
        ),
    )
    training_image: str | None = Field(
        default=None,
        description="Override entire GPU training image (registry/name:tag).",
    )
    tasks_image: str | None = Field(
        default=None,
        description="Override entire CPU tasks image (registry/name:tag).",
    )

    default_job_resource_cpu_request: str = Field(default="1")
    default_job_resource_memory_request: str = Field(default="8Gi")
    default_job_resource_cpu_limit: str = Field(default="4")
    default_job_resource_memory_limit: str = Field(default="16Gi")

    default_training_execution_profile: str = Field(
        default="gpu",
        description="Default single-node GPU profile when training.execution_profile is omitted.",
    )
    default_distributed_execution_profile: str = Field(
        default="gpu_distributed",
        description="Default multi-node (num_nodes>1) distributed-GPU execution profile.",
    )

    multinode_shared_storage_path: str | None = Field(
        default=None,
        description=(
            "Shared filesystem path (e.g. an NFS mount) used as BASE_LOG_DIR for Ray's "
            "cross-node ENDED/barrier coordination. REQUIRED for multi-node jobs "
            "(num_nodes>1); compile() fails fast when unset. Single-node jobs ignore it."
        ),
    )

    sandboxed_gym_default: bool = Field(
        default=True,
        description=(
            "When true, GRPO jobs with custom environment filesets compile with "
            "env.nemo_gym.sandboxed=true (platform operator default)."
        ),
    )

    gym_runtime_image: str | None = Field(
        default=None,
        description=(
            "Image the Gym sandbox runs for sandboxed GRPO (registry/name:tag). "
            "Defaults to the nmp-rl-training image, which already carries NeMo-Gym and the "
            "per-environment venvs."
        ),
    )

    sandbox_allow_internet: bool = Field(
        default=False,
        description=(
            "Sets env.nemo_gym.sandbox.allow_internet. NeMo-RL owns which DNS suffixes that "
            "opens; cluster-private, node-local and metadata destinations stay denied either "
            "way. Sandboxed environments build their server venvs at spin-up and cannot reach "
            "a package index without it, so enable this on clusters that run user-supplied "
            "environments and have no internal mirror."
        ),
    )

    sandbox_public_dns_allow: list[str] = Field(
        default_factory=list,
        description=(
            "Extra DNS suffixes or FQDNs added to env.nemo_gym.sandbox.network_policy."
            "public_dns_allow, consulted only when sandbox_allow_internet is true. NeMo-RL's "
            "built-in list covers *.com and *.org, so an environment whose package index lives "
            "on another TLD needs its host named here (e.g. hub.primeintellect.ai for Prime "
            "Intellect hub environments). Prefer exact FQDNs over wildcards. Operator-scoped: "
            "jobs cannot set it."
        ),
    )

    sandbox_rollout_chunk_size: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Rollouts carried by one POST to the Gym sandbox, as "
            "env.nemo_gym.sandbox.rollout_chunk_size. Leave unset to take NeMo-RL's default of "
            "8. A chunk's wall time scales with the tokens its rollouts ask for, and the "
            "OpenSandbox server caps how long one proxied request may stay open, so a job with "
            "a large max_new_tokens can need a smaller chunk: overrunning that cap fails every "
            "rollout with an HTTP 500 the job cannot recover from. The cap is fixed by the "
            "server build rather than exposed in its config, so the workable value is specific "
            "to this deployment and its models -- read the elapsed time on the failing POSTs "
            "and divide down from 8. Shrink this without raising sandbox_rollout_max_in_flight "
            "and step throughput falls with it: in-flight rollouts are the product of the two. "
            "Operator-scoped: jobs cannot set it."
        ),
    )

    sandbox_rollout_max_in_flight: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Concurrent POSTs to the Gym sandbox, as "
            "env.nemo_gym.sandbox.rollout_max_in_flight. Leave unset to take NeMo-RL's default "
            "of 8. In-flight rollouts are rollout_chunk_size × this value, so a deployment that "
            "lowers sandbox_rollout_chunk_size to stay under the proxy's per-request cap should "
            "raise this in proportion if it wants the same step throughput. Operator-scoped: "
            "jobs cannot set it."
        ),
    )

    sandbox_ttl_s: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Lifetime of the Gym sandbox pod, as env.nemo_gym.sandbox.ttl_s. Leave unset to "
            "take NeMo-RL's default of 14400 (4h). It is a reap timer for leaked sandboxes, "
            "not a job timeout, so it must exceed the longest training run the cluster "
            "accepts -- a GRPO job outliving it loses its sandbox mid-rollout and fails with "
            "a proxy 502. Cannot exceed the OpenSandbox server's max_sandbox_timeout_seconds "
            "(86400 as installed). Operator-scoped: jobs cannot set it."
        ),
    )

    sandbox_resources: dict[str, str] | None = Field(
        default_factory=lambda: {"cpu": "2", "memory": "4Gi"},
        description=(
            "CPU/memory requested for the Gym sandbox pod, as env.nemo_gym.sandbox.resources. "
            "Defaults to 2 CPU / 4Gi. The OpenSandbox SDK default (1 CPU / 2Gi) OOMKills a "
            "Gym server process per config entry plus its own Ray instance mid-rollout, "
            "surfacing as a proxy 502 rather than as a memory error. Raise further if the "
            "environment is larger. Operator-scoped: jobs cannot set it, since it spends "
            "cluster capacity."
        ),
    )

    job_storage_pvc_claim: str | None = Field(
        default=None,
        description=(
            "Name of the shared job-storage PersistentVolumeClaim (the same claim the Jobs "
            "controller mounts at the job storage path). Sandboxed GRPO re-mounts that claim "
            "into the Gym host so the sandbox can read the downloaded environment and dataset. "
            "Required before submitting sandboxed GRPO jobs. DPO and unsandboxed Gym can omit it "
            "even when platform.sandbox_cluster_capable is true."
        ),
    )


config = get_service_config(RlConfig)
platform_config = get_platform_config()

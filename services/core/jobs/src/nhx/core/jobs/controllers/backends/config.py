# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from nemo_helix_plugin.capabilities import probe_docker
from nhx.common.config import Runtime
from nhx.core.jobs.app.profiles import ExecutionProfileT
from nhx.core.jobs.controllers.backends.docker import DockerJobExecutionProfile, DockerJobExecutionProfileConfig
from nhx.core.jobs.controllers.backends.kubernetes import (
    KubernetesJobExecutionProfile,
    KubernetesJobExecutionProfileConfig,
    VolcanoJobExecutionProfile,
    VolcanoJobExecutionProfileConfig,
)
from nhx.core.jobs.controllers.backends.subprocess import (
    SubprocessJobExecutionProfile,
    SubprocessJobExecutionProfileConfig,
)
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_RUNTIME_NONE_UNSUPPORTED_BACKENDS = frozenset({"kubernetes_job", "volcano_job"})


class DefaultExecutionProfileConfig(BaseModel):
    """Holds default execution profile configurations for various backends."""

    # Note: these fields must match the name of the backend
    docker: DockerJobExecutionProfileConfig = Field(
        default_factory=DockerJobExecutionProfileConfig, description="Default Docker execution profile configuration"
    )
    kubernetes_job: KubernetesJobExecutionProfileConfig = Field(
        default_factory=KubernetesJobExecutionProfileConfig,
        description="Default Kubernetes execution profile configuration",
    )
    volcano_job: VolcanoJobExecutionProfileConfig = Field(
        default_factory=VolcanoJobExecutionProfileConfig, description="Default Volcano execution profile configuration"
    )
    subprocess: SubprocessJobExecutionProfileConfig = Field(
        default_factory=SubprocessJobExecutionProfileConfig,
        description="Default subprocess execution profile configuration",
    )


def get_default_executor_profiles_for_runtime(
    runtime: Runtime,
    defaults: DefaultExecutionProfileConfig,
    enable_subprocess_executor: bool | None = None,
) -> list:
    """Returns a list of default executor profiles based on the deployment runtime."""
    if enable_subprocess_executor is None:
        enable_subprocess_executor = runtime != Runtime.KUBERNETES

    logger.debug("Getting default executors for runtime: %s", runtime)
    executors = []
    if runtime == Runtime.DOCKER:
        executors.extend(
            [
                DockerJobExecutionProfile(
                    provider="cpu",
                    profile="default",
                    backend="docker",
                    config=defaults.docker,
                ),
                DockerJobExecutionProfile(
                    provider="gpu",
                    profile="default",
                    backend="docker",
                    config=defaults.docker,
                ),
            ]
        )
    elif runtime == Runtime.KUBERNETES:
        executors.extend(
            [
                KubernetesJobExecutionProfile(
                    provider="cpu",
                    profile="default",
                    backend="kubernetes_job",
                    config=defaults.kubernetes_job,
                ),
                KubernetesJobExecutionProfile(
                    provider="gpu",
                    profile="default",
                    backend="kubernetes_job",
                    config=defaults.kubernetes_job,
                ),
                # Customizer uses the gpu profile for both CPU preparation and
                # GPU training steps. The provider selects the requested resources.
                KubernetesJobExecutionProfile(
                    provider="cpu",
                    profile="gpu",
                    backend="kubernetes_job",
                    config=defaults.kubernetes_job,
                ),
                KubernetesJobExecutionProfile(
                    provider="gpu",
                    profile="gpu",
                    backend="kubernetes_job",
                    config=defaults.kubernetes_job,
                ),
                VolcanoJobExecutionProfile(
                    provider="gpu_distributed",
                    profile="default",
                    backend="volcano_job",
                    config=defaults.volcano_job,
                ),
            ]
        )

    if enable_subprocess_executor:
        executors.append(
            SubprocessJobExecutionProfile(
                provider="subprocess",
                profile="default",
                backend="subprocess",
                config=defaults.subprocess,
            )
        )

    if not executors:
        logger.warning(f"No default executors defined for runtime type: {runtime}")

    return executors


def merge_executor_profiles(
    custom_executors: list[ExecutionProfileT],
    default_executors: list[ExecutionProfileT],
    *,
    runtime: Runtime | None = None,
) -> list[ExecutionProfileT]:
    """
    Merge custom executor profiles with default profiles, giving precedence to custom profiles.

    If a custom profile has the same provider and profile as a default profile, the custom profile will override the default.
    If the custom profile matching a default profile has custom config values, those should override the default config values.
    When runtime is NONE, Kubernetes- and Volcano-backed custom profiles are skipped. Docker profiles (default or custom)
    are skipped whenever Docker is unavailable, independent of Runtime, so explicit Docker executors can still run in
    compose/test harnesses with a mounted socket.
    """

    merged_executors: dict[tuple[str, str], ExecutionProfileT] = {}
    docker_available: bool | None = None

    def _docker_is_available() -> bool:
        nonlocal docker_available
        if docker_available is None:
            # Uncached: this runs at jobs config import and must not pin the
            # process-wide probe cache before BackendRegistry.from_config's
            # authoritative boot probe.
            docker_available = probe_docker(use_cache=False).available
        return docker_available

    # Add default profiles first
    for executor in default_executors:
        if executor.backend == "docker" and not _docker_is_available():
            logger.warning(
                "Skipping executor profile %s/%s using backend '%s' because Docker is unavailable.",
                executor.provider,
                executor.profile,
                executor.backend,
            )
            continue
        merged_executors[(executor.provider, executor.profile)] = executor

    # Override with custom profiles
    for custom_executor in custom_executors:
        if runtime == Runtime.NONE and custom_executor.backend in _RUNTIME_NONE_UNSUPPORTED_BACKENDS:
            logger.warning(
                "Skipping executor profile %s/%s using backend '%s' because platform runtime is NONE.",
                custom_executor.provider,
                custom_executor.profile,
                custom_executor.backend,
            )
            continue
        if custom_executor.backend == "docker" and not _docker_is_available():
            logger.warning(
                "Skipping executor profile %s/%s using backend '%s' because Docker is unavailable.",
                custom_executor.provider,
                custom_executor.profile,
                custom_executor.backend,
            )
            continue

        key = (custom_executor.provider, custom_executor.profile)

        # If the custom executor matches a default, update the default config with custom values
        if key in merged_executors:
            default_executor = merged_executors[key]
            if custom_executor.backend != default_executor.backend or type(custom_executor.config) is not type(
                default_executor.config
            ):
                merged_executors[key] = custom_executor
                continue
            merged_config_data = default_executor.config.model_dump()
            merged_config_data.update(custom_executor.config.model_dump(exclude_unset=True))
            merged_config = type(default_executor.config)(**merged_config_data)
            merged_executors[key] = custom_executor.model_copy(update={"config": merged_config})
        # Otherwise, just add the custom executor
        else:
            merged_executors[key] = custom_executor

    return list(merged_executors.values())

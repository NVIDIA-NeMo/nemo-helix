# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from typing import Self, Sequence

from docker.errors import DockerException
from nemo_helix import NeMoHelix
from nemo_helix_plugin.capabilities import (
    CapabilityUnavailableError,
    probe_docker,
    reset_capability_cache,
)
from nhx.core.jobs.app.profiles import ExecutionProfileT
from nhx.core.jobs.app.schemas import BackendRef, ProfileRef, ProviderRef
from nhx.core.jobs.controllers.backends.base import DEFAULT_PROFILE, DEFAULT_PROVIDER, JobBackend
from nhx.core.jobs.controllers.backends.docker import CPUDockerJobBackend, GPUDockerJobBackend
from nhx.core.jobs.controllers.backends.kubernetes import (
    CPUKubernetesJobBackend,
    GPUKubernetesJobBackend,
    VolcanoJobBackend,
)
from nhx.core.jobs.controllers.backends.subprocess import SubprocessJobBackend
from nhx.core.jobs.controllers.backends.test import TestE2ECPUJobBackend, TestE2EGPUJobBackend
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

logger = logging.getLogger(__name__)

# Skip Docker backends only for daemon/connection failures after probe_docker()
# was True, or when init raises CapabilityUnavailableError. ValidationError and
# other programming/config errors must still fail startup.
_DOCKER_BACKEND_INIT_SKIPPABLE_ERRORS = (
    CapabilityUnavailableError,
    DockerException,
    RequestsConnectionError,
    RequestsTimeout,
    OSError,
)


@dataclass(frozen=True)
class RegistryKey:
    """Immutable key for identifying unique backend instances in the registry.

    Combines provider and execution profile to create a unique identifier
    for backend configurations.
    """

    provider: ProviderRef
    """The provider reference (e.g., "cpu", "gpu")"""
    profile: ProfileRef
    """The executor profile reference for this specific configuration (e.g., "default", ...)"""


@dataclass(frozen=True)
class BackendKey:
    """Immutable key for identifying unique backend instances in the registry.

    Combines provider and backend key to create a unique identifier
    for backend configurations.
    """

    provider: ProviderRef
    """The provider reference (e.g., "cpu", "gpu", "gpu_distributed")"""
    backend: BackendRef
    """The backend reference (e.g., "docker", "kubernetes", "volcano_job")"""


# Type alias for the backend registry mapping provider references to backend classes
BackendRegistryT = dict[BackendKey, type[JobBackend]]

# Global registry of available job backends mapped by provider name
backend_registry: BackendRegistryT = {
    BackendKey("cpu", "docker"): CPUDockerJobBackend,
    BackendKey("gpu", "docker"): GPUDockerJobBackend,
    BackendKey("cpu", "kubernetes_job"): CPUKubernetesJobBackend,
    BackendKey("gpu", "kubernetes_job"): GPUKubernetesJobBackend,
    BackendKey("gpu_distributed", "kubernetes_job"): GPUKubernetesJobBackend,
    BackendKey("gpu_distributed", "volcano_job"): VolcanoJobBackend,
    BackendKey("subprocess", "subprocess"): SubprocessJobBackend,
    BackendKey("cpu", "e2e"): TestE2ECPUJobBackend,
    BackendKey("gpu", "e2e"): TestE2EGPUJobBackend,
}


class BackendRegistry:
    """Registry for managing job execution backends across different providers and profiles.

    The BackendRegistry serves as a central repository for instantiated job backends,
    allowing the system to maintain separate backend instances for different execution
    profiles even when using the same underlying provider technology.

    Each backend instance is uniquely identified by a combination of provider and
    execution profile, enabling fine-grained control over job execution environments.
    """

    def __init__(self, registry: dict[RegistryKey, JobBackend]) -> None:
        """Initialize the registry with a pre-populated mapping of backends.

        Args:
            registry: Dictionary mapping RegistryKey instances to configured JobBackend instances
        """
        self._registry = registry

    @classmethod
    def from_config(
        cls,
        nhx_sdk: NeMoHelix,
        profiles: Sequence[ExecutionProfileT],
        backends: BackendRegistryT = backend_registry,
    ) -> Self:
        """Create a BackendRegistry from a list of execution profiles.

        This factory method processes execution profiles and instantiates the appropriate
        backend for each profile using the provided backend registry. Each execution
        profile's configuration is validated during backend instantiation.

        Args:
            profiles: List of execution profiles defining backend configurations
            backends: Registry of available backend classes (defaults to global backend_registry)

        Returns:
            A configured BackendRegistry instance with all backends initialized

        Raises:
            KeyError: If a profile references an executor not found in the backends registry
            ValidationError: If a profile's configuration is invalid for its backend type
        """
        registry: dict[RegistryKey, JobBackend] = {}
        docker_available: bool | None = None

        for executor in profiles:
            # Execution profiles are unique with respect to the provider
            # and profile combination
            registry_key = RegistryKey(executor.provider, executor.profile)
            backend_key = BackendKey(executor.provider, executor.backend)
            if backend_key not in backends:
                raise KeyError(
                    f"No backend registered for provider '{executor.provider}' and backend '{executor.backend}'"
                )
            backend = backends[backend_key]

            if executor.backend == "docker":
                # Process-lifetime cache is intentional: executor registration is
                # fixed at controller startup. Import-time merge uses
                # probe_docker(use_cache=False) so it does not pin this cache;
                # compile shares the cached boot verdict (restart required after
                # starting Docker mid-process).
                # Reset before the boot probe so an earlier transient miss (e.g.
                # deployments init probing before the daemon is ready) cannot
                # permanently skip Docker executors for this process.
                if docker_available is None:
                    reset_capability_cache()
                    docker_available = probe_docker().available
                if not docker_available:
                    logger.warning(
                        "Skipping job executor profile %s/%s using backend 'docker' because Docker is unavailable.",
                        executor.provider,
                        executor.profile,
                    )
                    continue

            # The config from the execution profile hasn't been validated
            # yet. Calling the backend constructor will serialize the raw
            # config into the backend's expected format and validate it
            try:
                registry[registry_key] = backend(nhx_sdk, executor.config, executor.profile)
            except _DOCKER_BACKEND_INIT_SKIPPABLE_ERRORS as exc:
                if executor.backend != "docker":
                    raise
                logger.warning(
                    "Skipping job executor profile %s/%s using backend 'docker' "
                    "because Docker backend initialization failed (%s).",
                    executor.provider,
                    executor.profile,
                    exc,
                )

        # Keep GET /v2/execution-profiles aligned with backends that registered.
        # ``profiles`` is built at import time and shared with the jobs API in local
        # standalone, so prune the list in place when callers pass a mutable list.
        if isinstance(profiles, list):
            registered = frozenset((key.provider, key.profile) for key in registry)
            kept = []
            for profile in profiles:
                if (profile.provider, profile.profile) in registered:
                    kept.append(profile)
            skipped = len(profiles) - len(kept)
            if skipped:
                logger.warning(
                    "Removed %s execution profile(s) that failed backend registration so advertised "
                    "profiles match the jobs controller registry.",
                    skipped,
                )
                profiles.clear()
                profiles.extend(kept)

        return cls(registry)

    def registered_profile_keys(self) -> frozenset[tuple[str, str]]:
        """Return (provider, profile) keys for backends that successfully registered."""
        return frozenset((key.provider, key.profile) for key in self._registry)

    def get_backend(self, *, provider: str | None = None, profile: str | None = None) -> JobBackend:
        """Retrieve a configured backend for the specified provider and profile.

        Args:
            provider: The provider identifier (e.g., "cpu", "gpu")
            profile: The execution profile identifier (e.g., "default", "a100", "test")

        Returns:
            The configured JobBackend instance for the specified provider and profile

        Raises:
            KeyError: If no backend is registered for the given provider and profile combination
        """
        if profile is None:
            profile = DEFAULT_PROFILE

        if provider is None:
            provider = DEFAULT_PROVIDER

        key = RegistryKey(provider, profile)
        if key in self._registry:
            return self._registry[key]

        raise KeyError(f"No backend found for provider '{provider}' and profile '{profile}'")

    def get_all_backends(self) -> list[JobBackend]:
        # sort by keys
        sorted_keys = sorted(list(self._registry.keys()), key=lambda k: f"{k.provider} {k.profile}")
        return [self._registry[k] for k in sorted_keys]

    def shutdown_all_backends(self) -> None:
        """Shutdown all registered backends in the registry."""
        for backend in self._registry.values():
            backend.shutdown()

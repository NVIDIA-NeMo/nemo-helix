# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RunnerBackendRegistry — factory that instantiates configured runner backends."""

from __future__ import annotations

import logging
import threading

from nemo_agents_plugin.config import AgentsConfig
from nemo_agents_plugin.entities import DeploymentMode, is_container_deployment_mode
from nemo_agents_plugin.runner.backend import RunnerBackend

logger = logging.getLogger(__name__)


class RunnerBackendRegistry:
    """Holds the subprocess and deployments-plugin runner backends.

    Subprocess mode uses :class:`~nemo_agents_plugin.runner.in_memory.InMemoryRunnerBackend`.
    Docker/k8s modes use :class:`~nemo_agents_plugin.runner.deployments_backend.DeploymentsRunnerBackend`.
    """

    def __init__(self, config: AgentsConfig) -> None:
        # Deferred: keep registry import cheap for CLI plugin discovery via the controller.
        from nemo_agents_plugin.runner.in_memory import InMemoryRunnerBackend

        self._config = config
        self._in_memory: RunnerBackend = InMemoryRunnerBackend(config.controller, host=config.subprocess_host)
        self._deployments: RunnerBackend | None = None
        logger.info(
            "Runner backend: InMemoryRunnerBackend (host=%s, port range start=%d)",
            config.subprocess_host,
            config.controller.port_range_start,
        )

    @property
    def backend(self) -> RunnerBackend:
        """Default (subprocess) backend — kept for callers that expect a single instance."""
        return self._in_memory

    def backend_for(self, mode: DeploymentMode) -> RunnerBackend:
        """Return the runner backend for *mode*."""
        if is_container_deployment_mode(mode):
            return self._deployments_backend()
        return self._in_memory

    def _deployments_backend(self) -> RunnerBackend:
        if self._deployments is None:
            # Deferred: deployments_backend pulls nemo-deployments + SDK machinery.
            from nemo_agents_plugin.runner.deployments_backend import DeploymentsRunnerBackend

            self._deployments = DeploymentsRunnerBackend(self._config)
            logger.info("Runner backend: DeploymentsRunnerBackend (lazy-init)")
        return self._deployments

    async def shutdown(self) -> None:
        await self._in_memory.shutdown()
        if self._deployments is not None:
            await self._deployments.shutdown()


_BACKEND_SINGLETON: RunnerBackend | None = None
_REGISTRY_SINGLETON: RunnerBackendRegistry | None = None
_BACKEND_LOCK = threading.Lock()


def set_runner_backend(backend: RunnerBackend) -> None:
    """Publish the process-wide default RunnerBackend (subprocess)."""
    global _BACKEND_SINGLETON  # noqa: PLW0603
    with _BACKEND_LOCK:
        _BACKEND_SINGLETON = backend


def set_runner_registry(registry: RunnerBackendRegistry) -> None:
    """Publish the process-wide registry (subprocess + deployments)."""
    global _REGISTRY_SINGLETON, _BACKEND_SINGLETON  # noqa: PLW0603
    with _BACKEND_LOCK:
        _REGISTRY_SINGLETON = registry
        _BACKEND_SINGLETON = registry.backend


def get_runner_backend() -> RunnerBackend:
    """Return the process-wide default RunnerBackend (lazy-builds for out-of-process callers)."""
    global _BACKEND_SINGLETON, _REGISTRY_SINGLETON  # noqa: PLW0603
    if _BACKEND_SINGLETON is None:
        with _BACKEND_LOCK:
            if _BACKEND_SINGLETON is None:
                registry = RunnerBackendRegistry(AgentsConfig.get())
                _REGISTRY_SINGLETON = registry
                _BACKEND_SINGLETON = registry.backend
    return _BACKEND_SINGLETON


def get_runner_registry() -> RunnerBackendRegistry:
    """Return the process-wide registry, creating it if needed."""
    global _REGISTRY_SINGLETON, _BACKEND_SINGLETON  # noqa: PLW0603
    if _REGISTRY_SINGLETON is None:
        with _BACKEND_LOCK:
            if _REGISTRY_SINGLETON is None:
                registry = RunnerBackendRegistry(AgentsConfig.get())
                _REGISTRY_SINGLETON = registry
                _BACKEND_SINGLETON = registry.backend
    return _REGISTRY_SINGLETON


def _reset_runner_backend_for_tests() -> None:
    global _BACKEND_SINGLETON, _REGISTRY_SINGLETON  # noqa: PLW0603
    with _BACKEND_LOCK:
        _BACKEND_SINGLETON = None
        _REGISTRY_SINGLETON = None

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named executor registry for deployment backends."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Self

from nemo_deployments_plugin.backends.base import DeploymentBackend, MissingBackendDependencyError
from nemo_deployments_plugin.backends.docker.backend import DockerDeploymentBackend
from nemo_deployments_plugin.backends.k8s.backend import K8sDeploymentBackend
from nemo_deployments_plugin.backends.openshell.backend import OpenShellDeploymentBackend
from nemo_helix import AsyncNeMoHelix

logger = logging.getLogger(__name__)

BACKEND_CLASSES: dict[str, type[DeploymentBackend]] = {
    "docker": DockerDeploymentBackend,
    "k8s": K8sDeploymentBackend,
    "openshell": OpenShellDeploymentBackend,
}


@dataclass(frozen=True)
class ExecutorSpec:
    name: str
    backend: str
    config: dict[str, Any]


@dataclass(frozen=True)
class UnavailableExecutor:
    """A configured executor that was skipped because its backend is unavailable."""

    backend: str
    reason: str


class ExecutorNotFoundError(KeyError):
    """Raised when no executor matches the requested name."""


class UnknownBackendTypeError(KeyError):
    """Raised when an executor references an unknown backend type."""


class ExecutorRegistry:
    """Maps executor names to configured DeploymentBackend singletons."""

    def __init__(
        self,
        executors: dict[str, DeploymentBackend],
        *,
        default_executor: str | None,
        unavailable: dict[str, UnavailableExecutor] | None = None,
    ) -> None:
        self._executors = executors
        self._default_executor = default_executor
        self._unavailable = unavailable or {}

    @classmethod
    def from_config(
        cls,
        sdk: AsyncNeMoHelix,
        specs: list[ExecutorSpec],
        *,
        default_executor: str | None = None,
        backend_classes: dict[str, type[DeploymentBackend]] | None = None,
    ) -> Self:
        classes = backend_classes if backend_classes is not None else BACKEND_CLASSES
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError("Duplicate executor names are not allowed.")
        executors: dict[str, DeploymentBackend] = {}
        unavailable: dict[str, UnavailableExecutor] = {}
        try:
            for spec in specs:
                if spec.backend not in classes:
                    raise UnknownBackendTypeError(f"Unknown backend type '{spec.backend}' for executor '{spec.name}'.")
                try:
                    executors[spec.name] = classes[spec.backend](sdk, spec.config)
                except MissingBackendDependencyError as exc:
                    # Capability missing (optional packaging extra, unreachable Docker
                    # daemon, etc.): skip just that executor so the deployments service
                    # can still boot. Remember it so resolving the name later explains
                    # the missing backend instead of looking like a typo. A configured
                    # default_executor that failed to register still fails fast below;
                    # silent clearing hid config mistakes and made debugging harder.
                    unavailable[spec.name] = UnavailableExecutor(backend=spec.backend, reason=str(exc))
                    logger.warning(
                        "Skipping executor '%s': backend '%s' is unavailable (%s)",
                        spec.name,
                        spec.backend,
                        exc,
                    )
            if default_executor and default_executor not in executors:
                default_spec = next((spec for spec in specs if spec.name == default_executor), None)
                backend_hint = (
                    f"backend '{default_spec.backend}'"
                    if default_spec is not None
                    else "its required backend capability"
                )
                raise ExecutorNotFoundError(
                    f"default_executor '{default_executor}' is not registered "
                    "(the backend is unavailable or the executor is missing from configuration). "
                    f"Configure a registered default_executor, or restore {backend_hint}."
                )
        except Exception:
            for backend in executors.values():
                backend.shutdown()
            raise
        return cls(executors, default_executor=default_executor, unavailable=unavailable)

    @classmethod
    def empty(cls) -> Self:
        """Registry with zero executors — valid at scaffold startup."""
        return cls({}, default_executor=None)

    def resolve(self, name: str | None = None) -> DeploymentBackend:
        executor_name = name or self._default_executor
        if executor_name is None:
            raise ExecutorNotFoundError("No executor specified and no default_executor configured.")
        if executor_name not in self._executors:
            skipped = self._unavailable.get(executor_name)
            if skipped is not None:
                raise ExecutorNotFoundError(
                    f"Executor '{executor_name}' is configured but its backend "
                    f"'{skipped.backend}' is unavailable: {skipped.reason}"
                )
            raise ExecutorNotFoundError(f"Executor '{executor_name}' is not registered.")
        return self._executors[executor_name]

    def shutdown_all(self) -> None:
        for name, backend in self._executors.items():
            logger.debug("Shutting down executor '%s'", name)
            backend.shutdown()

    def all_backends(self) -> list[DeploymentBackend]:
        return list(self._executors.values())

    def registered_names(self) -> list[str]:
        return list(self._executors.keys())

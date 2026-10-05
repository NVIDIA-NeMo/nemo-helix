# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tune backend discovery."""

from __future__ import annotations

import importlib.metadata
from functools import cache

from nemo_optimization.backends.protocol import OptimizationBackend, OptimizationPhase

OPTIMIZATION_BACKENDS_GROUP = "nemo.optimization.backends"


class OptimizationBackendDiscoveryError(RuntimeError):
    """Raised when Tune backend discovery fails."""


@cache
def discover_optimization_backends() -> dict[str, OptimizationBackend]:
    backends: dict[str, OptimizationBackend] = {}
    for entry in importlib.metadata.entry_points(group=OPTIMIZATION_BACKENDS_GROUP):
        try:
            backend_cls = entry.load()
        except Exception as exc:  # pragma: no cover - defensive
            raise OptimizationBackendDiscoveryError(f"Failed to load optimization backend {entry.name!r}") from exc
        if not isinstance(backend_cls, type):
            backend = backend_cls
        else:
            backend = backend_cls()
        if not isinstance(backend, OptimizationBackend):
            raise OptimizationBackendDiscoveryError(
                f"Optimization backend {entry.name!r} must implement OptimizationBackend"
            )
        backends[entry.name] = backend
    return backends


def discover_optimization_backends_for_phase(phase: OptimizationPhase) -> dict[str, OptimizationBackend]:
    """Return registered backends that advertise support for *phase*."""

    return {
        name: backend
        for name, backend in discover_optimization_backends().items()
        if backend.capabilities.supports(phase)
    }


def require_optimization_backend(name: str, *, phase: OptimizationPhase) -> OptimizationBackend:
    """Return a registered backend by name and phase, or raise a detailed error."""

    backends = discover_optimization_backends()
    backend = backends.get(name)
    if backend is None:
        raise OptimizationBackendDiscoveryError(
            f"Optimization backend {name!r} is not registered. Available backends: {sorted(backends)}"
        )
    if not backend.capabilities.supports(phase):
        phase_backends = discover_optimization_backends_for_phase(phase)
        raise OptimizationBackendDiscoveryError(
            f"Optimization backend {name!r} does not support the {phase.value!r} phase. "
            f"Available {phase.value} backends: {sorted(phase_backends)}"
        )
    return backend

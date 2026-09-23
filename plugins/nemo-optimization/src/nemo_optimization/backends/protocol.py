# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tune backend protocol."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from nemo_platform_plugin.client.adapter import SyncPlatformClient
from nemo_platform_plugin.job_context import JobContext


class OptimizationPhase(str, Enum):
    """Ordered optimization phases supported by registered backends."""

    NUMERIC = "numeric"
    PROMPT = "prompt"


class OptimizationPhaseStatus(str, Enum):
    """Status for one optimization phase result."""

    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


_RESERVED_RESULT_KEYS = frozenset({"status", "backend", "phase", "trial_number_range"})


@dataclass(frozen=True)
class OptimizationBackendCapabilities:
    """Phases a backend can validate and execute.

    Backends must advertise at least one phase and must not list a phase more
    than once. The router uses this declaration before invoking either backend
    method; implementations must apply the same check when called directly.
    """

    phases: tuple[OptimizationPhase, ...]

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("Optimization backend capabilities must declare at least one phase.")
        if any(not isinstance(phase, OptimizationPhase) for phase in self.phases):
            raise TypeError(f"Optimization backend capabilities require OptimizationPhase values: {self.phases!r}.")
        if len(set(self.phases)) != len(self.phases):
            raise ValueError(f"Optimization backend capabilities contain duplicate phases: {self.phases!r}.")

    def supports(self, phase: OptimizationPhase) -> bool:
        """Return whether the backend may validate and execute ``phase``."""

        return phase in self.phases


@dataclass(frozen=True)
class OptimizationPhaseRequest:
    """Input passed from the orchestrator to one optimization phase.

    ``trial_number_offset`` is the first global trial number available to this
    phase and therefore cannot be negative.
    """

    payload: dict[str, Any]
    phase: OptimizationPhase
    experiment_id: str | None = None
    trial_number_offset: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.phase, OptimizationPhase):
            raise TypeError(f"Optimization phase request requires an OptimizationPhase, got {self.phase!r}.")
        if isinstance(self.trial_number_offset, bool) or not isinstance(self.trial_number_offset, int):
            raise TypeError("Optimization phase trial_number_offset must be an integer.")
        if self.trial_number_offset < 0:
            raise ValueError("Optimization phase trial_number_offset must be non-negative.")


@dataclass(frozen=True)
class OptimizationPhaseResult:
    """Backend-agnostic result for one optimization phase.

    ``optimized_payload`` is intentionally kept in memory for phase handoff; job
    result serialization should publish sanitized artifacts rather than echoing
    full Fabric payloads that may contain credentials.

    ``trial_count`` is the number of trials actually executed, not the planned
    count. Together with the non-negative offset it defines the half-open global
    trial range returned by :attr:`trial_number_range`.
    """

    phase: OptimizationPhase
    backend: str
    status: OptimizationPhaseStatus
    optimized_payload: dict[str, Any]
    summary: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    trial_count: int = 0
    trial_number_offset: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.phase, OptimizationPhase):
            raise TypeError(f"Optimization phase result requires an OptimizationPhase, got {self.phase!r}.")
        if not isinstance(self.status, OptimizationPhaseStatus):
            raise TypeError(f"Optimization phase result requires an OptimizationPhaseStatus, got {self.status!r}.")
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("Optimization phase result backend must be a non-empty string.")
        if isinstance(self.trial_count, bool) or not isinstance(self.trial_count, int):
            raise TypeError("Optimization phase result trial_count must be an integer.")
        if self.trial_count < 0:
            raise ValueError("Optimization phase result trial_count must be non-negative.")
        if isinstance(self.trial_number_offset, bool) or not isinstance(self.trial_number_offset, int):
            raise TypeError("Optimization phase result trial_number_offset must be an integer.")
        if self.trial_number_offset < 0:
            raise ValueError("Optimization phase result trial_number_offset must be non-negative.")

    @property
    def trial_number_range(self) -> dict[str, int]:
        return {
            "start": self.trial_number_offset,
            "end_exclusive": self.trial_number_offset + self.trial_count,
            "count": self.trial_count,
        }

    def to_result_dict(self) -> dict[str, Any]:
        """Return a JSON-shaped summary suitable for the job result."""

        result = copy.deepcopy(dict(self.summary))
        artifact_keys = set(self.artifacts)
        if reserved := sorted(artifact_keys & _RESERVED_RESULT_KEYS):
            raise ValueError(f"Phase artifacts use reserved result field(s): {reserved}.")
        result.update(copy.deepcopy(dict(self.artifacts)))
        result.update(
            {
                "status": self.status.value,
                "backend": self.backend,
                "phase": self.phase.value,
                "trial_number_range": self.trial_number_range,
            }
        )
        return result


@runtime_checkable
class OptimizationBackend(Protocol):
    """Runtime-checkable contract implemented by optimization backends.

    ``validate_phase`` must reject unsupported phases and configuration errors
    before expensive work. ``run_phase`` must independently reject unsupported
    phases for callers that bypass validation, and its result must identify the
    requested phase and the backend's registered name.
    """

    name: str
    capabilities: OptimizationBackendCapabilities

    def validate_phase(
        self,
        request: OptimizationPhaseRequest,
        *,
        ctx: JobContext,
        sdk: SyncPlatformClient | None = None,
    ) -> None:
        """Validate a phase request before any expensive optimization work starts."""
        ...

    def run_phase(
        self,
        request: OptimizationPhaseRequest,
        *,
        ctx: JobContext,
        sdk: SyncPlatformClient | None = None,
    ) -> OptimizationPhaseResult:
        """Execute one optimizer phase for the given Fabric-native payload."""
        ...

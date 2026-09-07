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
    """Capabilities advertised by one registered optimization backend."""

    phases: tuple[OptimizationPhase, ...]

    def supports(self, phase: OptimizationPhase) -> bool:
        return phase in self.phases


@dataclass(frozen=True)
class OptimizationPhaseRequest:
    """Input passed from the orchestrator to one optimization phase."""

    payload: dict[str, Any]
    phase: OptimizationPhase
    experiment_id: str | None = None
    trial_number_offset: int = 0


@dataclass(frozen=True)
class OptimizationPhaseResult:
    """Backend-agnostic result for one optimization phase.

    ``optimized_payload`` is intentionally kept in memory for phase handoff; job
    result serialization should publish sanitized artifacts rather than echoing
    full Fabric payloads that may contain credentials.
    """

    phase: OptimizationPhase
    backend: str
    status: OptimizationPhaseStatus
    optimized_payload: dict[str, Any]
    summary: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    trial_count: int = 0
    trial_number_offset: int = 0

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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt GA backend stub."""

from __future__ import annotations

import copy
from typing import Any, ClassVar

from nemo_helix_plugin.client.adapter import SyncHelixClient
from nemo_helix_plugin.job_context import JobContext

from nemo_optimization.backends.protocol import (
    OptimizationBackendCapabilities,
    OptimizationPhase,
    OptimizationPhaseRequest,
    OptimizationPhaseResult,
    OptimizationPhaseStatus,
)
from nemo_optimization.optimizer_config import OptimizerConfigError, parse_optimizer_config


class GaBackendError(RuntimeError):
    """Raised when prompt GA is requested before the backend ships."""


class GaBackend:
    name: ClassVar[str] = "ga"
    capabilities: ClassVar[OptimizationBackendCapabilities] = OptimizationBackendCapabilities(
        phases=(OptimizationPhase.PROMPT,)
    )

    def run_study(
        self,
        payload: dict[str, Any],
        *,
        ctx: JobContext,
        sdk: SyncHelixClient | None = None,
    ) -> dict[str, Any]:
        request = OptimizationPhaseRequest(payload=payload, phase=OptimizationPhase.PROMPT)
        self.validate_phase(request, ctx=ctx, sdk=sdk)
        result = self.run_phase(
            request,
            ctx=ctx,
            sdk=sdk,
        )
        return result.to_result_dict()

    def validate_phase(
        self,
        request: OptimizationPhaseRequest,
        *,
        ctx: JobContext,
        sdk: SyncHelixClient | None = None,
    ) -> None:
        del ctx, sdk
        if not self.capabilities.supports(request.phase):
            raise OptimizerConfigError(
                f"GA backend {self.name!r} does not support phase {request.phase.value!r}; "
                f"supported phases: {[phase.value for phase in self.capabilities.phases]}.",
                phase=request.phase.value,
            )
        parse_optimizer_config(request.payload)

    def run_phase(
        self,
        request: OptimizationPhaseRequest,
        *,
        ctx: JobContext,
        sdk: SyncHelixClient | None = None,
    ) -> OptimizationPhaseResult:
        del sdk
        if not self.capabilities.supports(request.phase):
            raise GaBackendError(
                f"GA backend {self.name!r} does not support phase {request.phase.value!r}; "
                f"supported phases: {[phase.value for phase in self.capabilities.phases]}."
            )
        message = (
            "optimizer.prompt.enabled is not supported yet. "
            "Prompt GA is tracked separately and will be implemented in the GA algorithm stack."
        )
        del ctx
        return OptimizationPhaseResult(
            phase=OptimizationPhase.PROMPT,
            backend=self.name,
            status=OptimizationPhaseStatus.FAILED,
            optimized_payload=copy.deepcopy(request.payload),
            summary={"error": message},
            trial_count=0,
            trial_number_offset=request.trial_number_offset,
        )

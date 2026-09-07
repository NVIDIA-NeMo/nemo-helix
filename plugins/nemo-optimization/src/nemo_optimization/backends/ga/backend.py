# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt GA backend stub."""

from __future__ import annotations

import copy
from typing import Any, ClassVar

from nemo_platform_plugin.client.adapter import SyncPlatformClient
from nemo_platform_plugin.job_context import JobContext

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
        sdk: SyncPlatformClient | None = None,
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
        sdk: SyncPlatformClient | None = None,
    ) -> None:
        del ctx, sdk
        if request.phase is not OptimizationPhase.PROMPT:
            raise OptimizerConfigError(
                f"GA backend does not support the {request.phase.value!r} phase.", phase=request.phase.value
            )
        parse_optimizer_config(request.payload)

    def run_phase(
        self,
        request: OptimizationPhaseRequest,
        *,
        ctx: JobContext,
        sdk: SyncPlatformClient | None = None,
    ) -> OptimizationPhaseResult:
        del sdk
        if request.phase is not OptimizationPhase.PROMPT:
            raise GaBackendError(f"GA backend does not support the {request.phase.value!r} phase.")
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

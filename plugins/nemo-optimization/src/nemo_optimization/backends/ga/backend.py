# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt GA optimize backend."""

from __future__ import annotations

from typing import Any, ClassVar

from nemo_helix_plugin.client.adapter import SyncHelixClient
from nemo_helix_plugin.job_context import JobContext

from nemo_optimization.atif_metadata import resolve_experiment_id
from nemo_optimization.backends.ga.config import (
    GaConfigError,
    GaPromptOptimizerConfig,
    parse_ga_prompt_optimizer_config,
)
from nemo_optimization.backends.ga.driver import (
    GaPromptOptimizationResult,
    GaPromptOptimizerError,
    run_ga_prompt_optimization,
)
from nemo_optimization.backends.ga.transform import (
    ModelPromptTransformer,
    PromptTransformer,
    PromptTransformError,
)
from nemo_optimization.backends.protocol import (
    OptimizationBackendCapabilities,
    OptimizationPhase,
    OptimizationPhaseRequest,
    OptimizationPhaseResult,
    OptimizationPhaseStatus,
)
from nemo_optimization.candidate import CandidateEvaluationError, CandidateEvaluator
from nemo_optimization.config import generate_optimize_id
from nemo_optimization.fabric_evaluator import FabricCandidateEvaluator
from nemo_optimization.optimizer_config import OptimizerConfigError

RESULT_NAME = "optimizer_results"


class GaBackendError(RuntimeError):
    """Raised when prompt GA backend usage is invalid."""


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
        result = self.run_phase(
            OptimizationPhaseRequest(payload=payload, phase=OptimizationPhase.PROMPT),
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
        if not self.capabilities.supports(request.phase):
            raise OptimizerConfigError(
                f"GA backend {self.name!r} does not support phase {request.phase.value!r}; "
                f"supported phases: {[phase.value for phase in self.capabilities.phases]}.",
                phase=request.phase.value,
            )
        try:
            config = parse_ga_prompt_optimizer_config(request.payload)
            _validate_prompt_eval(request.payload)
            _build_prompt_transformer(
                request.payload,
                model_name=config.model,
                sdk=sdk,
                workspace=ctx.workspace,
            )
        except (GaConfigError, PromptTransformError) as exc:
            raise OptimizerConfigError(str(exc), phase="prompt") from exc

    def run_phase(
        self,
        request: OptimizationPhaseRequest,
        *,
        ctx: JobContext,
        sdk: SyncHelixClient | None = None,
    ) -> OptimizationPhaseResult:
        if not self.capabilities.supports(request.phase):
            raise GaBackendError(
                f"GA backend {self.name!r} does not support phase {request.phase.value!r}; "
                f"supported phases: {[phase.value for phase in self.capabilities.phases]}."
            )
        payload = request.payload
        output_dir = ctx.storage.persistent / "results" / RESULT_NAME

        experiment_id = request.experiment_id or resolve_experiment_id(payload, generate_id=generate_optimize_id)
        try:
            config = parse_ga_prompt_optimizer_config(payload)
            evaluator = _build_prompt_evaluator(
                payload,
                metric_names=config.metric_names,
                output_dir=output_dir,
                experiment_id=experiment_id,
            )
            transformer = _build_prompt_transformer(
                payload,
                model_name=config.model,
                sdk=sdk,
                workspace=ctx.workspace,
            )
            result = run_ga_prompt_optimization(
                payload,
                output_dir,
                evaluator,
                transformer,
                config=config,
                trial_number_offset=request.trial_number_offset,
            )
        except KeyError as exc:
            return _failed_prompt_phase_result(
                payload,
                backend=self.name,
                experiment_id=experiment_id,
                error=f"payload optimizer section is missing required key: {exc}",
                optimized_payload=None,
                trial_count=0,
                trial_number_offset=request.trial_number_offset,
            )
        except (CandidateEvaluationError, GaConfigError, GaPromptOptimizerError, PromptTransformError) as exc:
            return _failed_prompt_phase_result(
                payload,
                backend=self.name,
                experiment_id=experiment_id,
                error=str(exc),
                optimized_payload=getattr(exc, "optimized_payload", None),
                trial_count=getattr(exc, "trial_count", 0),
                trial_number_offset=request.trial_number_offset,
            )

        summary = _phase_summary(result, experiment_id=experiment_id, payload=payload, config=config)
        return OptimizationPhaseResult(
            phase=OptimizationPhase.PROMPT,
            backend=self.name,
            status=OptimizationPhaseStatus.COMPLETED,
            optimized_payload=result.optimized_payload,
            summary=summary,
            trial_count=result.executed_trials,
            trial_number_offset=request.trial_number_offset,
        )


def _phase_summary(
    result: GaPromptOptimizationResult,
    *,
    experiment_id: str,
    payload: dict[str, Any],
    config: GaPromptOptimizerConfig,
) -> dict[str, Any]:
    best = result.best_individual
    metadata = payload.get("metadata")
    return {
        "experiment_id": experiment_id,
        "population_size": config.population_size,
        "generations": config.generations,
        "generations_completed": result.generations_completed,
        "executed_trials": result.executed_trials,
        "best_individual": best.individual_id,
        "best_phase_trial_number": best.phase_trial_number,
        "best_global_trial_number": best.global_trial_number,
        "best_prompts": dict(best.prompts),
        "best_metrics": dict(best.aggregate_metrics),
        "best_fitness": best.fitness,
        "metric_names": list(result.metric_names),
        "agent": metadata.get("name") if isinstance(metadata, dict) else None,
    }


def _failed_prompt_phase_result(
    payload: dict[str, Any],
    *,
    backend: str,
    experiment_id: str,
    error: str,
    optimized_payload: dict[str, Any] | None,
    trial_count: int,
    trial_number_offset: int,
) -> OptimizationPhaseResult:
    return OptimizationPhaseResult(
        phase=OptimizationPhase.PROMPT,
        backend=backend,
        status=OptimizationPhaseStatus.FAILED,
        optimized_payload=optimized_payload if optimized_payload is not None else payload,
        summary={"experiment_id": experiment_id, "error": error, "executed_trials": trial_count},
        trial_count=trial_count,
        trial_number_offset=trial_number_offset,
    )


def _build_prompt_evaluator(
    payload: dict[str, Any],
    *,
    metric_names: tuple[str, ...],
    output_dir,
    experiment_id: str,
) -> CandidateEvaluator:
    _validate_prompt_eval(payload)
    return FabricCandidateEvaluator(
        payload=payload,
        metric_names=metric_names,
        output_dir=output_dir,
        experiment_id=experiment_id,
    )


def _validate_prompt_eval(payload: dict[str, Any]) -> None:
    if isinstance(payload.get("eval"), dict):
        return

    raise GaConfigError("Prompt GA optimization requires payload.eval; prompt-only means no numeric phase.")


def _build_prompt_transformer(
    payload: dict[str, Any],
    *,
    model_name: str,
    sdk: NeMoPlatform | None,
    workspace: str,
) -> PromptTransformer:
    return ModelPromptTransformer(sdk=sdk, workspace=workspace, payload=payload, model_name=model_name)

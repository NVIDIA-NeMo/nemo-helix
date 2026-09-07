# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route an optimization request to one registered backend."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from nemo_platform_plugin.client.adapter import SyncPlatformClient
from nemo_platform_plugin.job_context import JobContext

from nemo_optimization.artifact_utils import sanitize_config_for_artifact
from nemo_optimization.atif_metadata import resolve_experiment_id
from nemo_optimization.backends.protocol import (
    OptimizationBackend,
    OptimizationPhase,
    OptimizationPhaseRequest,
    OptimizationPhaseResult,
    OptimizationPhaseStatus,
)
from nemo_optimization.config import generate_optimize_id
from nemo_optimization.config_overlay import apply_suggestions
from nemo_optimization.fabric import build_optimize_payload, require_fabric_agent_config
from nemo_optimization.optimizer_config import OptimizerConfigError
from nemo_optimization.registry import OptimizationBackendDiscoveryError, require_optimization_backend

RESULT_NAME = "optimizer_results"


class OptimizeRouterError(RuntimeError):
    """Raised when optimize routing fails."""


@dataclass(frozen=True)
class _PhasePlan:
    phase: OptimizationPhase
    backend_name: str
    backend: OptimizationBackend


class OptimizeRouter:
    @staticmethod
    def dispatch(
        *,
        agent_config: dict[str, Any] | None,
        optimize_config: dict[str, Any],
        ctx: JobContext,
        sdk: SyncPlatformClient | None = None,
    ) -> dict[str, Any]:
        payload = build_optimize_payload(agent_config=agent_config, optimize_config=optimize_config)
        return _run_phases(payload, ctx=ctx, sdk=sdk)

    @staticmethod
    def dispatch_payload(
        payload: dict[str, Any],
        *,
        ctx: JobContext,
        sdk: SyncPlatformClient | None = None,
    ) -> dict[str, Any]:
        require_fabric_agent_config(payload, label="optimize payload")
        return _run_phases(payload, ctx=ctx, sdk=sdk)


def _run_phases(payload: dict[str, Any], *, ctx: JobContext, sdk: SyncPlatformClient | None) -> dict[str, Any]:
    plan = _phase_plan(payload)
    experiment_id = resolve_experiment_id(payload, generate_id=generate_optimize_id)
    output_dir = ctx.storage.persistent / "results" / RESULT_NAME

    request = OptimizationPhaseRequest(copy.deepcopy(payload), plan.phase, experiment_id)
    try:
        plan.backend.validate_phase(request, ctx=ctx, sdk=sdk)
    except OptimizerConfigError as exc:
        result = OptimizationPhaseResult(
            phase=plan.phase,
            backend=plan.backend_name,
            status=OptimizationPhaseStatus.FAILED,
            optimized_payload=copy.deepcopy(payload),
            summary={
                "experiment_id": experiment_id,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "executed_trials": 0,
            },
        )
    else:
        result = plan.backend.run_phase(request, ctx=ctx, sdk=sdk)

    _write_artifacts(output_dir, experiment_id, result)
    return _result(result, experiment_id=experiment_id, output_dir=output_dir, ctx=ctx)


def _phase_plan(payload: dict[str, Any]) -> _PhasePlan:
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise OptimizeRouterError("optimizer section must be a mapping.")

    plan: list[_PhasePlan] = []
    for phase, default_backend in (
        (OptimizationPhase.NUMERIC, "optuna"),
        (OptimizationPhase.PROMPT, "ga"),
    ):
        section = _phase_section(optimizer, phase.value)
        if section is None or not section["enabled"]:
            continue
        backend_name = _backend_name(section, default_backend)
        plan.append(_PhasePlan(phase, backend_name, _require_backend(backend_name, phase)))
    if not plan:
        raise OptimizeRouterError(
            "No Tune backend selected. Set optimizer.numeric.enabled: true or optimizer.prompt.enabled: true."
        )
    if len(plan) > 1:
        raise OptimizeRouterError("Only one optimization phase may be enabled per request.")
    return plan[0]


def _phase_section(optimizer: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    if name not in optimizer:
        return None
    raw = optimizer[name]
    if not isinstance(raw, Mapping):
        raise OptimizeRouterError(f"optimizer.{name} must be a mapping.")
    section = dict(raw)
    if not isinstance(section.get("enabled"), bool):
        raise OptimizeRouterError(f"optimizer.{name}.enabled must be a boolean.")
    return section


def _backend_name(section: Mapping[str, Any], default: str) -> str:
    value = section.get("backend", default)
    if not isinstance(value, str):
        raise OptimizeRouterError("Optimizer backend name must be a string.")
    if not value.strip():
        raise OptimizeRouterError("Optimizer backend name must not be empty.")
    return value.strip()


def _require_backend(name: str, phase: OptimizationPhase) -> OptimizationBackend:
    try:
        return require_optimization_backend(name, phase=phase)
    except OptimizationBackendDiscoveryError as exc:
        raise OptimizeRouterError(str(exc)) from exc


def _write_artifacts(
    output_dir: Path,
    experiment_id: str,
    result: OptimizationPhaseResult,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": result.status.value,
        "backend": result.backend,
        "phase": result.phase.value,
        "experiment_id": experiment_id,
        "trial_number_range": result.trial_number_range,
        "phases": [_phase_summary(result)],
    }
    _write_json(output_dir / "optimization_summary.json", summary)
    _write_json(output_dir / "phase_results.json", [_phase_detail(result)])

    suffix = "failure" if result.status is OptimizationPhaseStatus.FAILED else "summary"
    name = "study_summary.json" if result.phase is OptimizationPhase.NUMERIC and suffix == "summary" else None
    name = name or f"{result.phase.value}_phase_{suffix}.json"
    _write_json(
        output_dir / name,
        {
            "status": result.status.value,
            "backend": result.backend,
            "phase": result.phase.value,
            **dict(result.summary),
        },
    )

    if result.status is OptimizationPhaseStatus.COMPLETED:
        _write_payload(output_dir / "final_optimized_payload.json", result.optimized_payload)
        _write_config(output_dir / "final_optimized_config.yml", result.optimized_payload)


def _result(
    result: OptimizationPhaseResult,
    *,
    experiment_id: str,
    output_dir: Path,
    ctx: JobContext,
) -> dict[str, Any]:
    ref = ctx.results.save(RESULT_NAME, output_dir).model_dump(mode="json")
    return {**result.to_result_dict(), "experiment_id": experiment_id, "result": ref}


def _phase_summary(result: OptimizationPhaseResult) -> dict[str, Any]:
    return {
        "phase": result.phase.value,
        "backend": result.backend,
        "status": result.status.value,
        "trial_number_range": result.trial_number_range,
        "summary": copy.deepcopy(dict(result.summary)),
    }


def _phase_detail(result: OptimizationPhaseResult) -> dict[str, Any]:
    return {**_phase_summary(result), "artifacts": copy.deepcopy(dict(result.artifacts))}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    _write_json(path, sanitize_config_for_artifact(payload))


def _write_config(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(sanitize_config_for_artifact(apply_suggestions(payload, {})), sort_keys=False),
        encoding="utf-8",
    )

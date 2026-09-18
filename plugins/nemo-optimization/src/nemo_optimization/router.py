# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose registered numeric and prompt optimization backends."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from nemo_helix_plugin.client.adapter import SyncHelixClient
from nemo_helix_plugin.job_context import JobContext

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
        sdk: SyncHelixClient | None = None,
    ) -> dict[str, Any]:
        payload = build_optimize_payload(agent_config=agent_config, optimize_config=optimize_config)
        return _run_phases(payload, ctx=ctx, sdk=sdk)

    @staticmethod
    def dispatch_payload(
        payload: dict[str, Any],
        *,
        ctx: JobContext,
        sdk: SyncHelixClient | None = None,
    ) -> dict[str, Any]:
        require_fabric_agent_config(payload, label="optimize payload")
        return _run_phases(payload, ctx=ctx, sdk=sdk)


def _run_phases(payload: dict[str, Any], *, ctx: JobContext, sdk: SyncHelixClient | None) -> dict[str, Any]:
    plan = _phase_plan(payload)
    experiment_id = resolve_experiment_id(payload, generate_id=generate_optimize_id)
    output_dir = ctx.storage.persistent / "results" / RESULT_NAME

    preflight_failure = _preflight(plan, payload=payload, experiment_id=experiment_id, ctx=ctx, sdk=sdk)
    if preflight_failure is not None:
        _write_artifacts(output_dir, experiment_id, preflight_failure, planned=len(plan))
        return _result(preflight_failure, experiment_id=experiment_id, output_dir=output_dir, ctx=ctx)

    results: list[OptimizationPhaseResult] = []
    current_payload = copy.deepcopy(payload)
    offset = 0
    for index, item in enumerate(plan):
        result = item.backend.run_phase(
            OptimizationPhaseRequest(
                payload=current_payload,
                phase=item.phase,
                experiment_id=experiment_id,
                trial_number_offset=offset,
            ),
            ctx=ctx,
            sdk=sdk,
        )
        results.append(result)
        offset += result.trial_count
        current_payload = copy.deepcopy(result.optimized_payload)
        if result.status is not OptimizationPhaseStatus.COMPLETED:
            results.extend(_skipped(plan[index + 1 :], current_payload, offset, result.phase))
            break

    _write_artifacts(output_dir, experiment_id, results, planned=len(plan))
    return _result(results, experiment_id=experiment_id, output_dir=output_dir, ctx=ctx)


def _phase_plan(payload: dict[str, Any]) -> list[_PhasePlan]:
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
    if plan:
        return plan
    raise OptimizeRouterError(
        "No Tune backend selected. Set optimizer.numeric.enabled: true or optimizer.prompt.enabled: true."
    )


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


def _preflight(
    plan: list[_PhasePlan],
    *,
    payload: dict[str, Any],
    experiment_id: str,
    ctx: JobContext,
    sdk: NeMoPlatform | None,
) -> list[OptimizationPhaseResult] | None:
    for index, item in enumerate(plan):
        try:
            item.backend.validate_phase(
                OptimizationPhaseRequest(copy.deepcopy(payload), item.phase, experiment_id),
                ctx=ctx,
                sdk=sdk,
            )
        except OptimizerConfigError as exc:
            failure = OptimizationPhaseResult(
                phase=item.phase,
                backend=item.backend_name,
                status=OptimizationPhaseStatus.FAILED,
                optimized_payload=copy.deepcopy(payload),
                summary={
                    "experiment_id": experiment_id,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "executed_trials": 0,
                },
            )
            return [
                *_skipped(plan[:index], payload, 0, item.phase),
                failure,
                *_skipped(plan[index + 1 :], payload, 0, item.phase),
            ]
    return None


def _skipped(
    plan: list[_PhasePlan],
    payload: dict[str, Any],
    offset: int,
    failed_phase: OptimizationPhase,
) -> list[OptimizationPhaseResult]:
    return [
        OptimizationPhaseResult(
            phase=item.phase,
            backend=item.backend_name,
            status=OptimizationPhaseStatus.SKIPPED,
            optimized_payload=copy.deepcopy(payload),
            summary={"reason": f"Skipped because the {failed_phase.value!r} phase failed."},
            trial_number_offset=offset,
        )
        for item in plan
    ]


def _write_artifacts(
    output_dir: Path,
    experiment_id: str,
    results: list[OptimizationPhaseResult],
    *,
    planned: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    status = _overall_status(results)
    summary = {
        "status": status.value,
        "backend": "orchestrator" if planned > 1 else results[0].backend,
        "phase": "multi" if planned > 1 else results[0].phase.value,
        "experiment_id": experiment_id,
        "trial_number_range": _trial_range(results),
        "phases": [_phase_summary(result) for result in results],
    }
    _write_json(output_dir / "optimization_summary.json", summary)
    _write_json(output_dir / "phase_results.json", [_phase_detail(result) for result in results])

    for result in results:
        if result.status is OptimizationPhaseStatus.SKIPPED:
            continue
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

    numeric = next(
        (
            result
            for result in results
            if result.phase is OptimizationPhase.NUMERIC and result.status is OptimizationPhaseStatus.COMPLETED
        ),
        None,
    )
    if numeric is not None and planned > 1:
        _write_payload(output_dir / "intermediate_numeric_payload.json", numeric.optimized_payload)
        _write_config(output_dir / "intermediate_numeric_config.yml", numeric.optimized_payload)

    if status is OptimizationPhaseStatus.COMPLETED and len(results) == planned:
        _write_payload(output_dir / "final_optimized_payload.json", results[-1].optimized_payload)
        _write_config(output_dir / "final_optimized_config.yml", results[-1].optimized_payload)


def _result(
    results: list[OptimizationPhaseResult],
    *,
    experiment_id: str,
    output_dir: Path,
    ctx: JobContext,
) -> dict[str, Any]:
    ref = ctx.results.save(RESULT_NAME, output_dir).model_dump(mode="json")
    if len(results) == 1:
        return {**results[0].to_result_dict(), "experiment_id": experiment_id, "result": ref}
    return {
        "status": _overall_status(results).value,
        "backend": "orchestrator",
        "phase": "multi",
        "experiment_id": experiment_id,
        "trial_number_range": _trial_range(results),
        "phases": [_phase_summary(result) for result in results],
        "result": ref,
    }


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


def _overall_status(results: list[OptimizationPhaseResult]) -> OptimizationPhaseStatus:
    return (
        OptimizationPhaseStatus.COMPLETED
        if results and all(result.status is OptimizationPhaseStatus.COMPLETED for result in results)
        else OptimizationPhaseStatus.FAILED
    )


def _trial_range(results: list[OptimizationPhaseResult]) -> dict[str, int]:
    start = results[0].trial_number_offset if results else 0
    count = sum(result.trial_count for result in results)
    return {"start": start, "end_exclusive": start + count, "count": count}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    _write_json(path, sanitize_config_for_artifact(payload))


def _write_config(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(sanitize_config_for_artifact(apply_suggestions(payload, {})), sort_keys=False),
        encoding="utf-8",
    )

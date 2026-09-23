# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

import nemo_optimization.router as router_module
import pytest
from nemo_helix_plugin.job_context import JobContext
from nemo_optimization.backends.ga.backend import GaBackend, GaBackendError
from nemo_optimization.backends.optuna.backend import OptunaBackend
from nemo_optimization.backends.optuna.study_driver import StudyDriverError
from nemo_optimization.backends.protocol import (
    OptimizationBackend,
    OptimizationBackendCapabilities,
    OptimizationPhase,
    OptimizationPhaseRequest,
    OptimizationPhaseResult,
    OptimizationPhaseStatus,
)
from nemo_optimization.optimizer_config import OptimizerConfigError
from nemo_optimization.router import OptimizeRouter, OptimizeRouterError


def test_dispatch_routes_numeric_to_optuna_study(ctx: JobContext) -> None:
    payload = {
        "schema_version": "fabric.agent/v1alpha1",
        "metadata": {"name": "demo"},
        "optimizer": {
            "numeric": {"enabled": True, "n_trials": 2},
            "eval_metrics": {
                "average_score": {"direction": "maximize", "weight": 1.0},
            },
            "search_space": {
                "temperature": {
                    "type": "fabric",
                    "path": "models.default.temperature",
                    "values": [0.0, 0.2],
                },
            },
        },
    }
    result = OptimizeRouter.dispatch_payload(payload, ctx=ctx)
    assert result["status"] == "completed"
    assert result["backend"] == "optuna"
    assert result["phase"] == "numeric"
    assert result["n_trials"] == 2
    assert result["trial_number_range"] == {"start": 0, "end_exclusive": 2, "count": 2}

    out_dir = ctx.storage.persistent / "results" / "optimizer_results"
    summary = json.loads((out_dir / "study_summary.json").read_text(encoding="utf-8"))
    assert summary["backend"] == "optuna"
    debug = json.loads((out_dir / "study_debug.json").read_text(encoding="utf-8"))
    assert debug["n_trials"] == 2
    assert len(debug["trials"]) == 2
    assert {t["state"] for t in debug["trials"]} == {"COMPLETE"}
    assert (out_dir / "optimized_config.yml").is_file()


def test_dispatch_uses_executed_trial_count_for_phase_range(ctx: JobContext) -> None:
    payload = {
        "schema_version": "fabric.agent/v1alpha1",
        "metadata": {"name": "demo"},
        "optimizer": {
            "numeric": {"enabled": True, "n_trials": 20},
            "target": 0.5,
            "eval_metrics": {
                "average_score": {"direction": "maximize", "weight": 1.0},
            },
            "search_space": {
                "temperature": {
                    "type": "fabric",
                    "path": "models.default.temperature",
                    "values": [1.0, 2.0],
                },
            },
        },
    }

    result = OptimizeRouter.dispatch_payload(payload, ctx=ctx)

    assert result["n_trials"] == 20
    assert result["executed_trials"] == 1
    assert result["trial_number_range"] == {"start": 0, "end_exclusive": 1, "count": 1}


def test_dispatch_prompt_enabled_returns_failed_phase_result(ctx: JobContext) -> None:
    payload = {
        "schema_version": "fabric.agent/v1alpha1",
        "models": {"prompt_optimizer": {"provider": "openai", "model": "gpt-5-mini"}},
        "instructions": {"system": {"content": "Base prompt."}},
        "optimizer": {
            "prompt": {"enabled": True, "backend": "ga", "model": "prompt_optimizer"},
            "eval_metrics": {
                "average_score": {"direction": "maximize", "weight": 1.0},
            },
            "search_space": {
                "system_prompt": {
                    "type": "fabric",
                    "path": "instructions.system.content",
                    "is_prompt": True,
                    "purpose": "Answer accurately.",
                }
            },
        },
    }
    result = OptimizeRouter.dispatch_payload(payload, ctx=ctx)

    assert result["status"] == "failed"
    assert result["backend"] == "ga"
    assert result["phase"] == "prompt"
    assert "not supported yet" in result["error"]
    assert (ctx.storage.persistent / "results" / "optimizer_results" / "prompt_phase_failure.json").is_file()


@pytest.mark.parametrize(
    ("phase", "section"),
    [
        ("numeric", "enabled"),
        ("prompt", ["enabled"]),
    ],
)
def test_dispatch_rejects_non_mapping_phase_sections(ctx: JobContext, phase: str, section: Any) -> None:
    payload = {"schema_version": "fabric.agent/v1alpha1", "optimizer": {phase: section}}

    with pytest.raises(OptimizeRouterError, match=f"optimizer.{phase} must be a mapping"):
        OptimizeRouter.dispatch_payload(payload, ctx=ctx)


@pytest.mark.parametrize(
    ("phase", "section"),
    [
        ("numeric", {}),
        ("numeric", {"enabled": "false"}),
        ("prompt", {}),
        ("prompt", {"enabled": 1}),
    ],
)
def test_dispatch_rejects_non_boolean_enabled_values(ctx: JobContext, phase: str, section: dict[str, Any]) -> None:
    payload = {"schema_version": "fabric.agent/v1alpha1", "optimizer": {phase: section}}

    with pytest.raises(OptimizeRouterError, match=f"optimizer.{phase}.enabled must be a boolean"):
        OptimizeRouter.dispatch_payload(payload, ctx=ctx)


def test_dispatch_rejects_multiple_enabled_phases(ctx: JobContext) -> None:
    payload = {
        "schema_version": "fabric.agent/v1alpha1",
        "optimizer": {
            "numeric": {"enabled": True},
            "prompt": {"enabled": True},
        },
    }

    with pytest.raises(OptimizeRouterError, match="Only one optimization phase"):
        OptimizeRouter.dispatch_payload(payload, ctx=ctx)


def test_dispatch_prompt_backend_must_support_prompt_phase(ctx: JobContext) -> None:
    payload = {
        "schema_version": "fabric.agent/v1alpha1",
        "models": {"prompt_optimizer": {"provider": "openai", "model": "gpt-5-mini"}},
        "instructions": {"system": {"content": "Base prompt."}},
        "optimizer": {
            "prompt": {"enabled": True, "backend": "optuna", "model": "prompt_optimizer"},
            "search_space": {
                "system_prompt": {
                    "type": "fabric",
                    "path": "instructions.system.content",
                    "is_prompt": True,
                    "purpose": "Answer accurately.",
                }
            },
        },
    }
    with pytest.raises(OptimizeRouterError, match="does not support"):
        OptimizeRouter.dispatch_payload(payload, ctx=ctx)


def test_dispatch_rejects_non_string_backend_name(ctx: JobContext) -> None:
    payload = {
        "schema_version": "fabric.agent/v1alpha1",
        "optimizer": {
            "numeric": {"enabled": True, "backend": 123, "n_trials": 1},
            "eval_metrics": {
                "average_score": {"direction": "maximize", "weight": 1.0},
            },
            "search_space": {
                "temperature": {
                    "type": "fabric",
                    "path": "models.default.temperature",
                    "values": [0.0],
                },
            },
        },
    }

    with pytest.raises(OptimizeRouterError, match="must be a string"):
        OptimizeRouter.dispatch_payload(payload, ctx=ctx)


def test_dispatch_requires_enabled_backend(ctx: JobContext) -> None:
    payload = {"schema_version": "fabric.agent/v1alpha1", "optimizer": {}}
    with pytest.raises(OptimizeRouterError, match="No Tune backend selected"):
        OptimizeRouter.dispatch_payload(payload, ctx=ctx)


def test_phase_result_rejects_reserved_artifact_fields() -> None:
    result = OptimizationPhaseResult(
        phase=OptimizationPhase.PROMPT,
        backend="ga",
        status=OptimizationPhaseStatus.FAILED,
        optimized_payload={},
        artifacts={"status": {"path": "bad"}},
    )

    with pytest.raises(ValueError, match="reserved result field"):
        result.to_result_dict()


@pytest.mark.parametrize(
    ("factory", "phase", "run_error"),
    [
        (GaBackend, OptimizationPhase.NUMERIC, GaBackendError),
        (OptunaBackend, OptimizationPhase.PROMPT, StudyDriverError),
    ],
)
def test_backends_apply_advertised_capabilities_consistently(
    factory: type[GaBackend] | type[OptunaBackend],
    phase: OptimizationPhase,
    run_error: type[Exception],
    ctx: JobContext,
) -> None:
    backend = factory()
    request = OptimizationPhaseRequest(payload={}, phase=phase)

    with pytest.raises(OptimizerConfigError, match="supported phases"):
        backend.validate_phase(request, ctx=ctx)
    with pytest.raises(run_error, match="supported phases"):
        backend.run_phase(request, ctx=ctx)


def test_protocol_rejects_invalid_capabilities_and_trial_ranges() -> None:
    with pytest.raises(ValueError, match="at least one phase"):
        OptimizationBackendCapabilities(phases=())
    with pytest.raises(ValueError, match="duplicate phases"):
        OptimizationBackendCapabilities(phases=(OptimizationPhase.PROMPT, OptimizationPhase.PROMPT))
    with pytest.raises(ValueError, match="trial_number_offset"):
        OptimizationPhaseRequest(payload={}, phase=OptimizationPhase.PROMPT, trial_number_offset=-1)
    with pytest.raises(ValueError, match="trial_count"):
        OptimizationPhaseResult(
            phase=OptimizationPhase.PROMPT,
            backend="ga",
            status=OptimizationPhaseStatus.FAILED,
            optimized_payload={},
            trial_count=-1,
        )


@pytest.mark.parametrize(
    ("result_phase", "result_backend", "message"),
    [
        (OptimizationPhase.NUMERIC, "ga", "returned phase"),
        (OptimizationPhase.PROMPT, "not-ga", "mismatched backend name"),
    ],
)
def test_router_rejects_backend_results_that_violate_the_plan(
    monkeypatch: pytest.MonkeyPatch,
    ctx: JobContext,
    result_phase: OptimizationPhase,
    result_backend: str,
    message: str,
) -> None:
    class InvalidResultBackend:
        name = "ga"
        capabilities = OptimizationBackendCapabilities(phases=(OptimizationPhase.PROMPT,))

        def validate_phase(self, request: OptimizationPhaseRequest, *, ctx: JobContext, sdk=None) -> None:  # noqa: ANN001
            del request, ctx, sdk

        def run_phase(
            self,
            request: OptimizationPhaseRequest,
            *,
            ctx: JobContext,
            sdk=None,  # noqa: ANN001
        ) -> OptimizationPhaseResult:
            del request, ctx, sdk
            return OptimizationPhaseResult(
                phase=result_phase,
                backend=result_backend,
                status=OptimizationPhaseStatus.COMPLETED,
                optimized_payload={},
            )

    plan = router_module._PhasePlan(OptimizationPhase.PROMPT, "ga", InvalidResultBackend())
    monkeypatch.setattr(router_module, "_phase_plan", lambda payload: plan)

    with pytest.raises(OptimizeRouterError, match=message):
        router_module._run_phases({}, ctx=ctx, sdk=None)


def test_backend_protocol_accepts_phase_only_backend() -> None:
    class PhaseOnlyBackend:
        name = "prompt-test"
        capabilities = OptimizationBackendCapabilities(phases=(OptimizationPhase.PROMPT,))

        def run_phase(
            self,
            request: OptimizationPhaseRequest,
            *,
            ctx: JobContext,
            sdk=None,  # noqa: ANN001
        ) -> OptimizationPhaseResult:
            del request, ctx, sdk
            return OptimizationPhaseResult(
                phase=OptimizationPhase.PROMPT,
                backend=self.name,
                status=OptimizationPhaseStatus.COMPLETED,
                optimized_payload={},
            )

        def validate_phase(
            self,
            request: OptimizationPhaseRequest,
            *,
            ctx: JobContext,
            sdk=None,  # noqa: ANN001
        ) -> None:
            del request, ctx, sdk

    assert isinstance(PhaseOnlyBackend(), OptimizationBackend)

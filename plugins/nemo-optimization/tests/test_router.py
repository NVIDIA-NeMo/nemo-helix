# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from typing import Any

import nemo_optimization.router as router_module
import pytest
import yaml
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
from nemo_optimization.candidate import CandidateEvaluationError, CandidateEvaluationResult
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


def test_dispatch_prompt_enabled_without_eval_returns_failed_phase(ctx: JobContext) -> None:
    payload = {
        "schema_version": "fabric.agent/v1alpha1",
        "models": {"prompt_optimizer": {"provider": "openai", "model": "gpt-5-mini"}},
        "instructions": {"system": {"content": "Base prompt."}},
        "optimizer": {
            "prompt": {
                "enabled": True,
                "backend": "ga",
                "model": "prompt_optimizer",
                "population_size": 3,
                "generations": 1,
            },
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
    assert result["executed_trials"] == 0
    assert "requires payload.eval" in result["error"]
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


@pytest.mark.integration
def test_dispatch_integrates_real_optuna_then_ga_with_injected_dependencies(
    ctx: JobContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluators: dict[str, _RecordingCandidateEvaluator] = {}

    def _numeric_evaluator(payload: dict[str, Any], **kwargs: Any) -> _RecordingCandidateEvaluator:
        del payload
        evaluator = _RecordingCandidateEvaluator(kwargs["metric_names"])
        evaluators["numeric"] = evaluator
        return evaluator

    def _prompt_evaluator(payload: dict[str, Any], **kwargs: Any) -> _RecordingCandidateEvaluator:
        del payload
        evaluator = _RecordingCandidateEvaluator(kwargs["metric_names"])
        evaluators["prompt"] = evaluator
        return evaluator

    monkeypatch.setattr("nemo_optimization.backends.optuna.backend._build_trial_evaluator", _numeric_evaluator)
    monkeypatch.setattr("nemo_optimization.backends.ga.backend._build_prompt_evaluator", _prompt_evaluator)
    monkeypatch.setattr(
        "nemo_optimization.backends.ga.backend._build_prompt_transformer",
        lambda payload, *, model_name, sdk, workspace: _DeterministicPromptTransformer(),
    )

    result = OptimizeRouter.dispatch_payload(_real_backend_payload(), ctx=ctx)

    assert result["status"] == "completed"
    assert result["backend"] == "orchestrator"
    assert result["phase"] == "multi"
    assert result["trial_number_range"] == {"start": 0, "end_exclusive": 4, "count": 4}
    assert [phase["backend"] for phase in result["phases"]] == ["optuna", "ga"]
    assert result["phases"][0]["trial_number_range"] == {"start": 0, "end_exclusive": 2, "count": 2}
    assert result["phases"][1]["trial_number_range"] == {"start": 2, "end_exclusive": 4, "count": 2}
    assert [call["trial_number"] for call in evaluators["numeric"].calls] == [0, 1]
    assert [call["trial_number"] for call in evaluators["prompt"].calls] == [2, 3]
    assert {call["metadata"]["nemo.optimizer.global_trial_number"] for call in evaluators["prompt"].calls} == {2, 3}

    out_dir = ctx.storage.persistent / "results" / "optimizer_results"
    assert (out_dir / "config_prompt_trial_002.yml").is_file()
    assert (out_dir / "config_prompt_trial_003.yml").is_file()
    intermediate = yaml.safe_load((out_dir / "intermediate_numeric_config.yml").read_text(encoding="utf-8"))
    final = yaml.safe_load((out_dir / "final_optimized_config.yml").read_text(encoding="utf-8"))
    assert intermediate["models"]["default"]["temperature"] == 0.4
    assert final["models"]["default"]["temperature"] == 0.4
    assert "[system_prompt: mutation]" in final["instructions"]["system"]["content"]
    phase_results = json.loads((out_dir / "phase_results.json").read_text(encoding="utf-8"))
    assert [phase["status"] for phase in phase_results] == ["completed", "completed"]


@pytest.mark.integration
def test_dispatch_real_prompt_config_failure_skips_numeric_preflight(ctx: JobContext) -> None:
    payload = _real_backend_payload(prompt_overrides={"population_size": 0})

    result = OptimizeRouter.dispatch_payload(payload, ctx=ctx)

    assert result["status"] == "failed"
    assert result["trial_number_range"] == {"start": 0, "end_exclusive": 0, "count": 0}
    assert [phase["status"] for phase in result["phases"]] == ["skipped", "failed"]
    assert result["phases"][1]["summary"]["error_type"] == "OptimizerConfigError"

    out_dir = ctx.storage.persistent / "results" / "optimizer_results"
    assert not (out_dir / "intermediate_numeric_config.yml").exists()
    assert not (out_dir / "intermediate_numeric_payload.json").exists()
    assert (out_dir / "prompt_phase_failure.json").is_file()
    assert not (out_dir / "final_optimized_config.yml").exists()


@pytest.mark.integration
def test_dispatch_real_failed_numeric_reports_executed_trial_range(
    ctx: JobContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nemo_optimization.backends.ga.backend._build_prompt_transformer",
        lambda payload, *, model_name, sdk, workspace: _DeterministicPromptTransformer(),
    )
    monkeypatch.setattr(
        "nemo_optimization.backends.optuna.backend._build_trial_evaluator",
        lambda payload, **kwargs: _AlwaysFailCandidateEvaluator(),
    )

    result = OptimizeRouter.dispatch_payload(_real_backend_payload(), ctx=ctx)

    assert result["status"] == "failed"
    assert result["trial_number_range"] == {"start": 0, "end_exclusive": 2, "count": 2}
    assert result["phases"][0]["trial_number_range"] == {"start": 0, "end_exclusive": 2, "count": 2}
    assert result["phases"][0]["summary"]["executed_trials"] == 2
    assert result["phases"][1]["status"] == "skipped"
    assert result["phases"][1]["trial_number_range"] == {"start": 2, "end_exclusive": 2, "count": 0}


def test_dispatch_orchestrates_numeric_then_prompt_with_payload_handoff(
    ctx: JobContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, dict[str, Any]]] = []

    def _backend(name: str, *, phase: OptimizationPhase):  # noqa: ANN001
        return _FakeBackend(name=name, phase=phase, calls=calls)

    monkeypatch.setattr("nemo_optimization.router.require_optimization_backend", _backend)

    result = OptimizeRouter.dispatch_payload(_combined_payload(), ctx=ctx)

    assert result["status"] == "completed"
    assert result["backend"] == "orchestrator"
    assert result["phase"] == "multi"
    assert result["experiment_id"] == "exp-router"
    assert result["trial_number_range"] == {"start": 0, "end_exclusive": 5, "count": 5}
    assert [(phase, offset) for phase, offset, _payload in calls] == [("numeric", 0), ("prompt", 2)]
    assert calls[1][2]["models"]["default"]["temperature"] == 0.4
    assert result["phases"][0]["trial_number_range"] == {"start": 0, "end_exclusive": 2, "count": 2}
    assert result["phases"][1]["trial_number_range"] == {"start": 2, "end_exclusive": 5, "count": 3}

    out_dir = ctx.storage.persistent / "results" / "optimizer_results"
    assert (out_dir / "optimization_summary.json").is_file()
    assert (out_dir / "phase_results.json").is_file()
    assert (out_dir / "intermediate_numeric_config.yml").is_file()
    assert (out_dir / "intermediate_numeric_payload.json").is_file()
    assert (out_dir / "final_optimized_config.yml").is_file()
    assert (out_dir / "final_optimized_payload.json").is_file()
    summary = json.loads((out_dir / "optimization_summary.json").read_text(encoding="utf-8"))
    assert [phase["phase"] for phase in summary["phases"]] == ["numeric", "prompt"]


def test_dispatch_stops_after_failed_prompt_but_keeps_numeric_artifacts(
    ctx: JobContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, dict[str, Any]]] = []

    def _backend(name: str, *, phase: OptimizationPhase):  # noqa: ANN001
        return _FakeBackend(name=name, phase=phase, calls=calls, fail_prompt=True)

    monkeypatch.setattr("nemo_optimization.router.require_optimization_backend", _backend)

    result = OptimizeRouter.dispatch_payload(_combined_payload(), ctx=ctx)

    assert result["status"] == "failed"
    assert result["trial_number_range"] == {"start": 0, "end_exclusive": 3, "count": 3}
    assert [phase["status"] for phase in result["phases"]] == ["completed", "failed"]
    assert [(phase, offset) for phase, offset, _payload in calls] == [("numeric", 0), ("prompt", 2)]

    out_dir = ctx.storage.persistent / "results" / "optimizer_results"
    assert (out_dir / "intermediate_numeric_config.yml").is_file()
    assert (out_dir / "intermediate_numeric_payload.json").is_file()
    assert not (out_dir / "final_optimized_config.yml").exists()
    summary = json.loads((out_dir / "optimization_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"


def test_dispatch_skips_prompt_after_failed_numeric(
    ctx: JobContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, dict[str, Any]]] = []

    def _backend(name: str, *, phase: OptimizationPhase):  # noqa: ANN001
        return _FakeBackend(name=name, phase=phase, calls=calls, fail_numeric=True)

    monkeypatch.setattr("nemo_optimization.router.require_optimization_backend", _backend)

    result = OptimizeRouter.dispatch_payload(_combined_payload(), ctx=ctx)

    assert result["status"] == "failed"
    assert result["trial_number_range"] == {"start": 0, "end_exclusive": 1, "count": 1}
    assert [(phase, offset) for phase, offset, _payload in calls] == [("numeric", 0)]
    assert [phase["status"] for phase in result["phases"]] == ["failed", "skipped"]
    assert result["phases"][1]["trial_number_range"] == {"start": 1, "end_exclusive": 1, "count": 0}


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


class _FakeBackend:
    def __init__(
        self,
        *,
        name: str,
        phase: OptimizationPhase,
        calls: list[tuple[str, int, dict[str, Any]]],
        fail_numeric: bool = False,
        fail_prompt: bool = False,
    ) -> None:
        self.name = name
        self.capabilities = OptimizationBackendCapabilities(phases=(phase,))
        self._phase = phase
        self._calls = calls
        self._fail_numeric = fail_numeric
        self._fail_prompt = fail_prompt

    def run_phase(
        self,
        request: OptimizationPhaseRequest,
        *,
        ctx: JobContext,
        sdk=None,  # noqa: ANN001
    ) -> OptimizationPhaseResult:
        del ctx, sdk
        self._calls.append((request.phase.value, request.trial_number_offset, copy.deepcopy(request.payload)))
        if self._phase is OptimizationPhase.NUMERIC:
            payload = copy.deepcopy(request.payload)
            payload.setdefault("models", {}).setdefault("default", {})["temperature"] = 0.4
            if self._fail_numeric:
                return OptimizationPhaseResult(
                    phase=OptimizationPhase.NUMERIC,
                    backend=self.name,
                    status=OptimizationPhaseStatus.FAILED,
                    optimized_payload=payload,
                    summary={"error": "numeric failed"},
                    trial_count=1,
                    trial_number_offset=request.trial_number_offset,
                )
            return OptimizationPhaseResult(
                phase=OptimizationPhase.NUMERIC,
                backend=self.name,
                status=OptimizationPhaseStatus.COMPLETED,
                optimized_payload=payload,
                summary={"best_params": {"temperature": 0.4}},
                trial_count=2,
                trial_number_offset=request.trial_number_offset,
            )

        payload = copy.deepcopy(request.payload)
        payload["instructions"]["system"]["content"] = "Tuned prompt."
        if self._fail_prompt:
            return OptimizationPhaseResult(
                phase=OptimizationPhase.PROMPT,
                backend=self.name,
                status=OptimizationPhaseStatus.FAILED,
                optimized_payload=copy.deepcopy(request.payload),
                summary={"error": "prompt failed"},
                trial_count=1,
                trial_number_offset=request.trial_number_offset,
            )
        return OptimizationPhaseResult(
            phase=OptimizationPhase.PROMPT,
            backend=self.name,
            status=OptimizationPhaseStatus.COMPLETED,
            optimized_payload=payload,
            summary={"best_prompts": {"system_prompt": "Tuned prompt."}},
            trial_count=3,
            trial_number_offset=request.trial_number_offset,
        )

    def validate_phase(
        self,
        request: OptimizationPhaseRequest,
        *,
        ctx: JobContext,
        sdk=None,  # noqa: ANN001
    ) -> None:
        del request, ctx, sdk


class _RecordingCandidateEvaluator:
    def __init__(self, metric_names: tuple[str, ...]) -> None:
        self._metric_names = metric_names
        self.calls: list[dict[str, Any]] = []

    def evaluate(
        self,
        *,
        trial_number: int,
        suggestions: dict[str, Any],
        trial_overlay: dict[str, Any],
        rep: int,
    ) -> CandidateEvaluationResult:
        self.calls.append(
            {
                "trial_number": trial_number,
                "suggestions": dict(suggestions),
                "metadata": copy.deepcopy(trial_overlay.get("metadata", {})),
                "rep": rep,
            }
        )
        if "models.default.temperature" in suggestions:
            score = float(suggestions["models.default.temperature"])
        else:
            score = float(len(str(suggestions["instructions.system.content"])))
        return CandidateEvaluationResult(aggregate_metrics={name: score for name in self._metric_names})


class _AlwaysFailCandidateEvaluator:
    def evaluate(
        self,
        *,
        trial_number: int,
        suggestions: dict[str, Any],
        trial_overlay: dict[str, Any],
        rep: int,
    ) -> CandidateEvaluationResult:
        del trial_number, suggestions, trial_overlay, rep
        raise CandidateEvaluationError("simulated evaluator failure")


class _DeterministicPromptTransformer:
    def mutate(
        self,
        *,
        prompt_name: str,
        prompt: str,
        purpose: str,
        prompt_format: str | None,
        feedback: str | None,
    ) -> str:
        del purpose, prompt_format, feedback
        return f"{prompt}\n\n[{prompt_name}: mutation]"

    def recombine(
        self,
        *,
        prompt_name: str,
        parent_a: str,
        parent_b: str,
        purpose: str,
        prompt_format: str | None,
        feedback: str | None,
    ) -> str:
        del prompt_name, purpose, prompt_format, feedback
        return f"{parent_a}\n\n{parent_b}"


def _real_backend_payload(*, prompt_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    prompt = {
        "enabled": True,
        "backend": "ga",
        "model": "prompt_optimizer",
        "population_size": 2,
        "generations": 1,
        "elitism": 0,
        "seed": 11,
    }
    if prompt_overrides:
        prompt.update(prompt_overrides)
    return {
        "schema_version": "fabric.agent/v1alpha1",
        "metadata": {"name": "real-backend-demo"},
        "eval": {"tasks": []},
        "models": {
            "default": {"provider": "openai", "model": "agent-model", "temperature": 0.0},
            "prompt_optimizer": {
                "provider": "openai",
                "model": "gpt-5-mini",
                "base_url": "https://example.test/v1",
                "api_key_env": "NVIDIA_API_KEY",
            },
        },
        "instructions": {"system": {"content": "Base prompt."}},
        "optimizer": {
            "experiment_id": "exp-real",
            "numeric": {"enabled": True, "backend": "optuna", "sampler": "grid", "n_trials": 2},
            "prompt": prompt,
            "reps_per_param_set": 1,
            "eval_metrics": {"average_score": {"direction": "maximize", "weight": 1.0}},
            "search_space": {
                "temperature": {
                    "type": "fabric",
                    "path": "models.default.temperature",
                    "values": [0.0, 0.4],
                },
                "system_prompt": {
                    "type": "fabric",
                    "path": "instructions.system.content",
                    "is_prompt": True,
                    "purpose": "Answer accurately.",
                },
            },
        },
    }


def _combined_payload() -> dict[str, Any]:
    return {
        "schema_version": "fabric.agent/v1alpha1",
        "metadata": {"name": "demo"},
        "models": {
            "default": {"provider": "openai", "model": "agent-model", "temperature": 0.0},
            "prompt_optimizer": {
                "provider": "openai",
                "model": "gpt-5-mini",
                "base_url": "https://example.test/v1",
                "api_key_env": "NVIDIA_API_KEY",
            },
        },
        "instructions": {"system": {"content": "Base prompt."}},
        "optimizer": {
            "experiment_id": "exp-router",
            "numeric": {"enabled": True, "backend": "optuna", "n_trials": 2},
            "prompt": {"enabled": True, "backend": "ga", "model": "prompt_optimizer"},
            "eval_metrics": {"average_score": {"direction": "maximize", "weight": 1.0}},
            "search_space": {
                "temperature": {
                    "type": "fabric",
                    "path": "models.default.temperature",
                    "values": [0.0, 0.4],
                },
                "system_prompt": {
                    "type": "fabric",
                    "path": "instructions.system.content",
                    "is_prompt": True,
                    "purpose": "Answer accurately.",
                },
            },
        },
    }

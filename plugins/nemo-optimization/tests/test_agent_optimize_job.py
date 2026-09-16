# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from typing import Any, cast

import pytest
from nemo_optimization.jobs import agent_optimize as job_module
from nemo_optimization.jobs.agent_optimize import NatAgentOptimizeJob
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.run_dependencies import LocalRunError


def _source() -> dict[str, Any]:
    return {
        "config_format": "nemo-agents-spec-v1",
        "name": "my-agent",
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes", "model": {"provider": "openai", "model": "m", "temperature": 0.0}}},
    }


def _config(**search_space: dict[str, Any]) -> dict[str, Any]:
    # A real optimizer.numeric config also declares search_space (see e.g.
    # examples/hermes-optimize/optimize-chatonly.yaml); the preserved
    # _optimizer_problems check (ported from the old validate_config) requires it.
    return {
        "optimizer": {
            "numeric": {"enabled": True},
            "search_space": search_space
            or {"temperature": {"type": "fabric", "path": "models.default.temperature", "values": [0.0, 0.2]}},
        }
    }


def test_optimize_runs_preflight_then_staging_then_dispatch_then_overlays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    monkeypatch.setattr(
        job_module,
        "preflight_validate_llm_models",
        lambda *a, **k: order.append("preflight"),
    )

    @contextlib.contextmanager
    def fake_staged(optimize_config, *, workspace, ctx, sdk):
        order.append("staged")
        yield {**optimize_config, "staged": True}

    monkeypatch.setattr(job_module, "_staged_dataset", fake_staged)

    def fake_dispatch(*, agent_config, optimize_config, ctx, sdk=None):
        order.append("dispatch")
        assert optimize_config["staged"] is True
        return {"best_params_by_path": {"models.default.temperature": 0.7}}

    monkeypatch.setattr(job_module.OptimizeRouter, "dispatch", staticmethod(fake_dispatch))

    optimized = NatAgentOptimizeJob().optimize(
        source_agent_config=_source(),
        config=_config(),
        ctx=cast(JobContext, object()),
        workspace="my-ws",
        sdk=None,
    )

    assert order == ["preflight", "staged", "dispatch"]
    assert optimized["harnesses"]["hermes"]["model"]["temperature"] == 0.7
    # The caller's mapping is an input, not scratch space.
    source = _source()
    assert source["harnesses"]["hermes"]["model"]["temperature"] == 0.0


def test_the_job_declares_its_strategy() -> None:
    assert NatAgentOptimizeJob.strategy == "nat"


def test_the_job_declares_its_task_module() -> None:
    assert NatAgentOptimizeJob.task_module == "nemo_optimization.tasks.agent_optimize"


def test_optimize_rejects_a_config_missing_an_optimizer_section(monkeypatch: pytest.MonkeyPatch) -> None:
    """The old ``NatOptimizationStrategy.validate_config`` behavior must survive the migration."""
    with pytest.raises(ValueError, match="optimizer section"):
        NatAgentOptimizeJob().optimize(
            source_agent_config=_source(),
            config={},
            ctx=cast(JobContext, object()),
            workspace="my-ws",
            sdk=None,
        )


def test_optimize_refuses_to_report_success_when_the_study_found_no_tuned_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(job_module, "preflight_validate_llm_models", lambda *a, **k: None)

    @contextlib.contextmanager
    def fake_staged(optimize_config, *, workspace, ctx, sdk):
        yield optimize_config

    monkeypatch.setattr(job_module, "_staged_dataset", fake_staged)
    monkeypatch.setattr(job_module.OptimizeRouter, "dispatch", staticmethod(lambda **kwargs: {"status": "completed"}))

    with pytest.raises(LocalRunError, match="nothing"):
        NatAgentOptimizeJob().optimize(
            source_agent_config=_source(),
            config=_config(),
            ctx=cast(JobContext, object()),
            workspace="my-ws",
            sdk=None,
        )


def _never_runs(name: str) -> Any:
    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"{name} must not run once the search space is refused")

    return _fail


def _refuse(monkeypatch: pytest.MonkeyPatch, search_space: dict[str, Any]) -> str:
    """Run optimize with *search_space* and return the refusal message.

    Preflight and dispatch are wired to explode: an untunable path has to be caught
    before the study spends a single model call.
    """
    monkeypatch.setattr(job_module, "preflight_validate_llm_models", _never_runs("preflight"))
    monkeypatch.setattr(job_module, "_staged_dataset", _never_runs("dataset staging"))
    monkeypatch.setattr(job_module.OptimizeRouter, "dispatch", staticmethod(_never_runs("the study")))

    with pytest.raises(LocalRunError) as excinfo:
        NatAgentOptimizeJob().optimize(
            source_agent_config=_source(),
            config=_config(**search_space),
            ctx=cast(JobContext, object()),
            workspace="my-ws",
            sdk=None,
        )
    return str(excinfo.value)


def test_a_spec_shaped_search_space_path_is_refused_before_the_study_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``harnesses.<name>.settings.<field>`` reads like a real location and is one — on the
    *spec*. The study mutates the Fabric package, whose harness key is 'harness', so every
    trial would run identically and the winner's tuned value would never have been measured.
    """
    message = _refuse(
        monkeypatch,
        {"max_tokens": {"type": "fabric", "path": "harnesses.hermes.settings.max_tokens", "values": [256, 512]}},
    )
    assert "harnesses.hermes.settings.max_tokens" in message
    assert "not a Fabric key" in message


def test_a_model_leaf_that_cannot_round_trip_is_refused_before_the_study_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal that the shipped MCP example used to hit only after four paid trials."""
    message = _refuse(
        monkeypatch,
        {"top_p": {"type": "fabric", "path": "models.default.top_p", "values": [0.9, 1.0]}},
    )
    assert "models.default.top_p" in message
    assert "round-trips" in message


def test_every_offending_path_is_named_in_one_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    message = _refuse(
        monkeypatch,
        {
            "top_p": {"type": "fabric", "path": "models.default.top_p", "values": [0.9, 1.0]},
            "nowhere": {"type": "fabric", "path": "llms.nowhere.temperature", "values": [0.1, 0.2]},
        },
    )
    assert "models.default.top_p" in message
    assert "llms.nowhere.temperature" in message


def test_a_fabric_shaped_harness_setting_is_accepted_and_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """The counterpart the study really can tune: the path passes validation and the
    overlay writes it back onto the spec's named harness."""
    source = _source()
    source["harnesses"]["hermes"]["settings"] = {"max_tokens": 256}

    monkeypatch.setattr(job_module, "preflight_validate_llm_models", lambda *a, **k: None)

    @contextlib.contextmanager
    def fake_staged(optimize_config, *, workspace, ctx, sdk):
        yield optimize_config

    monkeypatch.setattr(job_module, "_staged_dataset", fake_staged)
    monkeypatch.setattr(
        job_module.OptimizeRouter,
        "dispatch",
        staticmethod(lambda **kwargs: {"best_params_by_path": {"harness.settings.max_tokens": 512}}),
    )

    optimized = NatAgentOptimizeJob().optimize(
        source_agent_config=source,
        config=_config(
            max_tokens={"type": "fabric", "path": "harness.settings.max_tokens", "values": [256, 512]},
        ),
        ctx=cast(JobContext, object()),
        workspace="my-ws",
        sdk=None,
    )

    assert optimized["harnesses"]["hermes"]["settings"]["max_tokens"] == 512

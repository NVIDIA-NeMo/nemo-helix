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


def _config() -> dict[str, Any]:
    # A real optimizer.numeric config also declares search_space (see e.g.
    # examples/hermes-optimize/optimize-chatonly.yaml); the preserved
    # _optimizer_problems check (ported from the old validate_config) requires it.
    return {
        "optimizer": {
            "numeric": {"enabled": True},
            "search_space": {"temperature": {"type": "fabric", "path": "models.default.temperature"}},
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

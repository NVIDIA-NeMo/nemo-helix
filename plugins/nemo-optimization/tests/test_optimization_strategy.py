# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_optimization.optimization_strategy import NatOptimizationStrategy


def test_name_is_nat() -> None:
    assert NatOptimizationStrategy().name == "nat"


def test_validate_config_rejects_missing_optimizer_section() -> None:
    strategy = NatOptimizationStrategy()
    with pytest.raises(ValueError, match="optimizer section"):
        strategy.validate_config({"schema_version": "fabric.agent/v1alpha1"}, agent=None)


def test_validate_config_accepts_valid_fabric_config_with_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = NatOptimizationStrategy()
    monkeypatch.setattr("nemo_optimization.optimization_strategy._agent_problems", lambda config, *, agent: iter(()))
    # ``search_space`` is a sibling of ``numeric`` under ``optimizer``, matching the shape
    # ``_optimizer_problems`` reads and the examples under ``examples/hermes-optimize/``.
    strategy.validate_config(
        {"optimizer": {"numeric": {"enabled": True}, "search_space": {"lr": {}}}},
        agent="my-agent",
    )


def test_run_dispatches_to_optimize_router(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = NatOptimizationStrategy()
    captured = {}

    def _fake_dispatch(*, agent_config, optimize_config, ctx, sdk):
        captured.update(agent_config=agent_config, optimize_config=optimize_config)
        return {"status": "completed"}

    monkeypatch.setattr("nemo_optimization.optimization_strategy.OptimizeRouter.dispatch", staticmethod(_fake_dispatch))
    monkeypatch.setattr("nemo_optimization.optimization_strategy.preflight_validate_llm_models", lambda *a, **k: None)

    import contextlib

    @contextlib.contextmanager
    def _fake_staged_dataset(optimize_config, *, workspace, ctx, sdk):
        yield optimize_config

    monkeypatch.setattr("nemo_optimization.optimization_strategy._staged_dataset", _fake_staged_dataset)

    result = strategy.run(
        agent_config={"schema_version": "fabric.agent/v1alpha1"},
        source_agent_config=None,
        config={"optimizer": {"numeric": {"enabled": True}}},
        ctx=object(),
        workspace="default",
        sdk=None,
    )
    assert result == {"status": "completed"}
    assert captured["agent_config"] == {"schema_version": "fabric.agent/v1alpha1"}

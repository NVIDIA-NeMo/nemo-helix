# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib.metadata import EntryPoint
from typing import Any

import pytest
from nemo_agent_optimization_plugin.strategies import (
    OPTIMIZATION_STRATEGY_GROUP,
    OptimizationStrategy,
    OptimizationStrategyDiscoveryError,
    discover_optimization_strategies,
)


class _FakeStrategy:
    name = "fake"

    def validate_config(self, config: dict[str, Any], *, agent: str | None) -> None:
        del config, agent

    def run(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"status": "ok"}


class _Mismatched(_FakeStrategy):
    name = "other-name"


def test_discover_optimization_strategies_loads_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = EntryPoint(name="fake", value="test_strategies:_FakeStrategy", group=OPTIMIZATION_STRATEGY_GROUP)
    monkeypatch.setattr(
        "nemo_agent_optimization_plugin.strategies.importlib.metadata.entry_points",
        lambda group: [entry] if group == OPTIMIZATION_STRATEGY_GROUP else [],
    )
    discover_optimization_strategies.cache_clear()
    strategies = discover_optimization_strategies()
    assert set(strategies) == {"fake"}
    assert isinstance(strategies["fake"], OptimizationStrategy)


def test_discover_optimization_strategies_rejects_name_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = EntryPoint(name="fake", value="test_strategies:_Mismatched", group=OPTIMIZATION_STRATEGY_GROUP)
    monkeypatch.setattr(
        "nemo_agent_optimization_plugin.strategies.importlib.metadata.entry_points",
        lambda group: [entry] if group == OPTIMIZATION_STRATEGY_GROUP else [],
    )
    discover_optimization_strategies.cache_clear()
    with pytest.raises(OptimizationStrategyDiscoveryError, match="loaded a strategy named"):
        discover_optimization_strategies()

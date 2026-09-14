# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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


class _NotAStrategy:
    """Lacks the OptimizationStrategy protocol surface entirely."""


class _StubEntryPoint:
    """Stands in for importlib.metadata.EntryPoint: discovery only uses .name and .load()."""

    def __init__(self, name: str, loads: object) -> None:
        self.name = name
        self._loads = loads

    def load(self) -> object:
        if isinstance(self._loads, Exception):
            raise self._loads
        return self._loads


def test_discover_optimization_strategies_loads_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _StubEntryPoint("fake", _FakeStrategy)
    monkeypatch.setattr(
        "nemo_agent_optimization_plugin.strategies.importlib.metadata.entry_points",
        lambda group: [entry] if group == OPTIMIZATION_STRATEGY_GROUP else [],
    )
    discover_optimization_strategies.cache_clear()
    strategies = discover_optimization_strategies()
    assert set(strategies) == {"fake"}
    assert isinstance(strategies["fake"], OptimizationStrategy)


def test_discover_optimization_strategies_rejects_name_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _StubEntryPoint("fake", _Mismatched)
    monkeypatch.setattr(
        "nemo_agent_optimization_plugin.strategies.importlib.metadata.entry_points",
        lambda group: [entry] if group == OPTIMIZATION_STRATEGY_GROUP else [],
    )
    discover_optimization_strategies.cache_clear()
    with pytest.raises(OptimizationStrategyDiscoveryError, match="loaded a strategy named"):
        discover_optimization_strategies()


def test_discover_optimization_strategies_wraps_load_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _StubEntryPoint("fake", ImportError("boom"))
    monkeypatch.setattr(
        "nemo_agent_optimization_plugin.strategies.importlib.metadata.entry_points",
        lambda group: [entry] if group == OPTIMIZATION_STRATEGY_GROUP else [],
    )
    discover_optimization_strategies.cache_clear()
    with pytest.raises(OptimizationStrategyDiscoveryError, match="Failed to load"):
        discover_optimization_strategies()


def test_discover_optimization_strategies_rejects_non_protocol_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _StubEntryPoint("fake", _NotAStrategy)
    monkeypatch.setattr(
        "nemo_agent_optimization_plugin.strategies.importlib.metadata.entry_points",
        lambda group: [entry] if group == OPTIMIZATION_STRATEGY_GROUP else [],
    )
    discover_optimization_strategies.cache_clear()
    with pytest.raises(OptimizationStrategyDiscoveryError, match="must implement OptimizationStrategy"):
        discover_optimization_strategies()

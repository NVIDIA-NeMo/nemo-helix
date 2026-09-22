# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strategy discovery: a class variable, not a base class or an entry-point group."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from nemo_agent_optimization_plugin import discovery
from nemo_agent_optimization_plugin.discovery import discover_strategies, discover_strategy_jobs
from nemo_agent_optimization_plugin.schemas.strategies import OptimizationStrategy
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext


class _Strategy(NemoJob):
    name: ClassVar[str] = "optimize"
    description: ClassVar[str] = "The job's own description, which the listing does not use."
    nemo_agent_optimization_strategy: ClassVar[OptimizationStrategy] = OptimizationStrategy(
        name="fake", description="What the strategy optimizes."
    )

    def run(self, config: dict, *, ctx: JobContext) -> dict[str, Any]:
        return {}


class _OtherStrategy(NemoJob):
    name: ClassVar[str] = "tune"
    nemo_agent_optimization_strategy: ClassVar[OptimizationStrategy] = OptimizationStrategy(name="fake")

    def run(self, config: dict, *, ctx: JobContext) -> dict[str, Any]:
        return {}


class _Unrelated(NemoJob):
    name: ClassVar[str] = "unrelated"

    def run(self, config: dict, *, ctx: JobContext) -> dict[str, Any]:
        return {}


class _LateStrategy(NemoJob):
    """A second, differently named strategy — sorts after ``fake``."""

    name: ClassVar[str] = "tune"
    nemo_agent_optimization_strategy: ClassVar[OptimizationStrategy] = OptimizationStrategy(name="acme")

    def run(self, config: dict, *, ctx: JobContext) -> dict[str, Any]:
        return {}


class _MistypedStrategy(NemoJob):
    """Declares the variable, but as the bare name the contract used to take."""

    name: ClassVar[str] = "mistyped"
    nemo_agent_optimization_strategy: ClassVar[str] = "fake"

    def run(self, config: dict, *, ctx: JobContext) -> dict[str, Any]:
        return {}


class _BlankStrategy(NemoJob):
    name: ClassVar[str] = "blank"
    nemo_agent_optimization_strategy: ClassVar[OptimizationStrategy] = OptimizationStrategy(name="")

    def run(self, config: dict, *, ctx: JobContext) -> dict[str, Any]:
        return {}


def test_keys_strategies_by_their_declared_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discovery, "discover_jobs", lambda: {"whatever.optimize": _Strategy})
    assert discover_strategy_jobs() == {"fake": _Strategy}


def test_the_entry_point_key_does_not_matter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strategies share no key convention — the class variable is the whole signal."""
    monkeypatch.setattr(
        discovery,
        "discover_jobs",
        lambda: {"a-plugin.anything-at-all": _Strategy, "b-plugin.unrelated": _Unrelated},
    )
    assert discover_strategy_jobs() == {"fake": _Strategy}


def test_ignores_jobs_that_declare_no_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        discovery,
        "discover_jobs",
        lambda: {"other.unrelated": _Unrelated, "other.blank": _BlankStrategy},
    )
    assert discover_strategy_jobs() == {}


def test_a_bare_string_is_not_a_strategy_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    """The declaration carries a description too, so only the model counts as one."""
    monkeypatch.setattr(discovery, "discover_jobs", lambda: {"other.mistyped": _MistypedStrategy})
    assert discover_strategy_jobs() == {}


def test_a_duplicate_strategy_name_keeps_the_first_and_warns(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        discovery,
        "discover_jobs",
        lambda: {"a.optimize": _Strategy, "b.tune": _OtherStrategy},
    )
    with caplog.at_level("WARNING"):
        assert discover_strategy_jobs() == {"fake": _Strategy}
    assert "claimed by both" in caplog.text


def test_strategies_are_listed_as_their_plugin_declared_them(monkeypatch: pytest.MonkeyPatch) -> None:
    """The description comes from the declaration, not from the job it hangs on."""
    monkeypatch.setattr(discovery, "discover_jobs", lambda: {"whatever.optimize": _Strategy})

    assert discover_strategies() == [OptimizationStrategy(name="fake", description="What the strategy optimizes.")]


def test_strategies_are_sorted_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        discovery,
        "discover_jobs",
        lambda: {"a.optimize": _Strategy, "b.tune": _LateStrategy},
    )

    assert [strategy.name for strategy in discover_strategies()] == ["acme", "fake"]

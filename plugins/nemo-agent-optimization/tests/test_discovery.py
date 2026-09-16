# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_agent_optimization_plugin import discovery
from nemo_agent_optimization_plugin.discovery import (
    AgentOptimizeDiscoveryError,
    discover_agent_optimize_jobs,
)
from nemo_agent_optimization_plugin.job_base import AgentOptimizeJob
from nemo_platform_plugin.job import NemoJob


class _Fake(AgentOptimizeJob):
    name = "agent_optimize"
    strategy = "fake"

    def optimize(self, **kwargs):
        return {}


class _Unrelated(NemoJob):
    name = "unrelated"

    def run(self, config, *, ctx):
        return {}


class _NoStrategy(AgentOptimizeJob):
    name = "agent_optimize"

    def optimize(self, **kwargs):
        return {}


def test_finds_agent_optimize_subclasses_keyed_by_strategy(monkeypatch) -> None:
    monkeypatch.setattr(discovery, "discover_jobs", lambda: {"opt.agent_optimize": _Fake})
    assert discover_agent_optimize_jobs() == {"fake": _Fake}


def test_ignores_jobs_that_are_not_agent_optimize_jobs(monkeypatch) -> None:
    monkeypatch.setattr(
        discovery, "discover_jobs",
        lambda: {"opt.agent_optimize": _Fake, "other.unrelated": _Unrelated},
    )
    assert discover_agent_optimize_jobs() == {"fake": _Fake}


def test_excludes_the_base_class_itself(monkeypatch) -> None:
    monkeypatch.setattr(discovery, "discover_jobs", lambda: {"x.base": AgentOptimizeJob})
    assert discover_agent_optimize_jobs() == {}


def test_a_subclass_without_a_strategy_fails_loudly(monkeypatch) -> None:
    monkeypatch.setattr(discovery, "discover_jobs", lambda: {"opt.agent_optimize": _NoStrategy})
    with pytest.raises(AgentOptimizeDiscoveryError, match="declares no 'strategy'"):
        discover_agent_optimize_jobs()

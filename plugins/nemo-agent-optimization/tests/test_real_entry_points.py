# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve every agent-optimize strategy from real, installed entry points.

Every other test touching :func:`discover_agent_optimize_jobs` monkeypatches
``discover_jobs``, so a typo or stale path in any strategy plugin's
``pyproject.toml`` ``[project.entry-points."nemo.jobs"]`` table would never
surface as a test failure. This test calls the real, unpatched
``discover_agent_optimize_jobs()`` so it actually resolves the ``nat``,
``prompt-master``, ``switchyard``, and ``experimentalist`` entry points
installed in this environment.
"""

from __future__ import annotations

import importlib.util

import pytest
from nemo_agent_optimization_plugin.discovery import discover_agent_optimize_jobs
from nemo_agent_optimization_plugin.job_base import AgentOptimizeJob

# Strategy name -> a module owned by the plugin that registers it, used only
# to decide whether the plugin is installed in this venv at all.
_REQUIRED_STRATEGY_MODULES = {
    "nat": "nemo_optimization",
    "prompt-master": "prompt_master_plugin",
    "switchyard": "nemo_switchyard",
    "experimentalist": "nemo_experimentalist_plugin",
}

_MISSING_PLUGINS = sorted(
    strategy for strategy, module in _REQUIRED_STRATEGY_MODULES.items() if importlib.util.find_spec(module) is None
)


@pytest.mark.skipif(
    bool(_MISSING_PLUGINS),
    reason=(
        "Not all agent-optimize strategy plugins are installed in this venv, so this test "
        f"cannot exercise their real entry points. Missing plugins for strategies: {_MISSING_PLUGINS}. "
        "Run `uv sync` from the repo root (this is a single workspace covering all four plugins) "
        "and re-run."
    ),
)
def test_all_four_strategies_resolve_from_real_entry_points() -> None:
    """discover_agent_optimize_jobs(), unpatched, must resolve all four strategies."""
    strategies = discover_agent_optimize_jobs()

    for strategy in _REQUIRED_STRATEGY_MODULES:
        assert strategy in strategies, (
            f"Expected strategy {strategy!r} in discover_agent_optimize_jobs(), got {sorted(strategies)}. "
            'Check the owning plugin\'s pyproject.toml [project.entry-points."nemo.jobs"] table for a '
            "stale or misspelled 'module:ClassName' target."
        )
        job_cls = strategies[strategy]
        assert isinstance(job_cls, type) and issubclass(job_cls, AgentOptimizeJob), (
            f"Strategy {strategy!r} resolved to {job_cls!r}, which is not an AgentOptimizeJob subclass."
        )

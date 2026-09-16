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

That test alone does not guard the bundled wrapper manifest
(``packages/nemo_platform/pyproject.toml``): that package is a permanent
``[tool.uv.workspace]`` member, so both the standalone plugin distribution
and the bundled wrapper distribution register the same ``*.agent_optimize``
entry-point name in any dev/CI environment, and ``entry_points()`` lookup
dedups same-named entries with no guaranteed winner. A typo introduced only
in the wrapper manifest could be masked by the correct standalone entry and
the test above would still pass. The second test here reads the wrapper
manifest directly as data (no entry-point resolution involved), so a typo
there fails regardless of what any installed distribution registers.
"""

from __future__ import annotations

import importlib
import importlib.util
import tomllib
from pathlib import Path

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

# plugins/nemo-agent-optimization/tests/test_real_entry_points.py -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_WRAPPER_PYPROJECT = _REPO_ROOT / "packages" / "nemo_platform" / "pyproject.toml"

_EXPECTED_AGENT_OPTIMIZE_KEYS = {
    "optimization.agent_optimize",
    "prompt-master.agent_optimize",
    "switchyard.agent_optimize",
    "experimentalist.agent_optimize",
}


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


@pytest.mark.skipif(
    bool(_MISSING_PLUGINS),
    reason=(
        "Not all agent-optimize strategy plugins are installed in this venv, so this test "
        f"cannot import the classes the wrapper manifest points at. Missing plugins for "
        f"strategies: {_MISSING_PLUGINS}. Run `uv sync` from the repo root and re-run."
    ),
)
def test_bundled_wrapper_manifest_declares_all_four_agent_optimize_entries() -> None:
    """Read packages/nemo_platform/pyproject.toml as data and check it directly.

    This does not go through entry-point resolution at all, so it catches a typo
    in the bundled wrapper manifest even when a same-named, correct entry from the
    standalone plugin distribution would otherwise mask it during discovery.
    """
    assert _WRAPPER_PYPROJECT.is_file(), f"Expected bundled wrapper manifest at {_WRAPPER_PYPROJECT}"

    with _WRAPPER_PYPROJECT.open("rb") as f:
        pyproject = tomllib.load(f)

    entry_points = pyproject["project"]["entry-points"]
    assert "nemo.optimization-strategy" not in entry_points, (
        "The retired 'nemo.optimization-strategy' entry-point group must not reappear in "
        f"{_WRAPPER_PYPROJECT} — agent-optimize strategies are now nemo.jobs entries, and a "
        "known `make vendor` generator bug silently preserves this table once it exists."
    )

    jobs = entry_points["nemo.jobs"]
    agent_optimize_keys = {key for key in jobs if key.endswith(".agent_optimize")}
    assert agent_optimize_keys == _EXPECTED_AGENT_OPTIMIZE_KEYS, (
        f"Expected exactly {_EXPECTED_AGENT_OPTIMIZE_KEYS} in {_WRAPPER_PYPROJECT}'s "
        f'[project.entry-points."nemo.jobs"], got {agent_optimize_keys}.'
    )

    for key in sorted(agent_optimize_keys):
        target = jobs[key]
        module_name, _, class_name = target.partition(":")
        assert module_name and class_name, f"Malformed entry-point target for {key!r}: {target!r}"
        module = importlib.import_module(module_name)
        job_cls = getattr(module, class_name)
        assert isinstance(job_cls, type) and issubclass(job_cls, AgentOptimizeJob), (
            f"{key!r} -> {target!r} does not resolve to an AgentOptimizeJob subclass."
        )

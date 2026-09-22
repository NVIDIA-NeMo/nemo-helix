# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the real, installed entry points.

Every other test here monkeypatches :func:`discover_strategy_jobs`, so a typo or
stale path in a strategy plugin's ``[project.entry-points."nemo.jobs"]`` table
would never surface as a failure. These tests call the unpatched discovery and
read the bundled wrapper manifest as data.

The wrapper manifest (``packages/nemo_helix/pyproject.toml``) needs its own
check: it is a permanent ``[tool.uv.workspace]`` member, so in any dev/CI venv
both it and the standalone plugin distribution register the same entry-point
names, and ``entry_points()`` dedups same-named entries with no guaranteed
winner. A typo introduced only in the wrapper would be masked by the correct
standalone entry.
"""

from __future__ import annotations

import importlib
import importlib.util
import tomllib
from pathlib import Path

import pytest
from nemo_agent_optimization_plugin.cli import OPTIMIZE_CLI_GROUP
from nemo_agent_optimization_plugin.discovery import STRATEGY_ATTR, declared_strategy, discover_strategy_jobs
from nemo_agent_optimization_plugin.jobs.run_strategy import RunStrategyJob
from nemo_helix_plugin.discovery import discover
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.scheduler import submit_path_for

# Strategy name -> a module owned by the plugin that ships it, used only to decide
# whether that plugin is installed in this venv at all.
_REQUIRED_STRATEGY_MODULES = {
    "nat": "nemo_optimization",
}

_MISSING_PLUGINS = sorted(
    strategy for strategy, module in _REQUIRED_STRATEGY_MODULES.items() if importlib.util.find_spec(module) is None
)

# plugins/nemo-agent-optimization/tests/test_real_entry_points.py -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_WRAPPER_PYPROJECT = _REPO_ROOT / "packages" / "nemo_helix" / "pyproject.toml"

_ROUTER_JOB_KEY = "agent-optimization.run-strategy"
_STRATEGY_JOB_KEYS = {"agent-optimization.optimize"}

_STRATEGIES_INSTALLED = pytest.mark.skipif(
    bool(_MISSING_PLUGINS),
    reason=(
        "Not every optimization strategy plugin is installed in this venv, so this test cannot "
        f"exercise its real entry points. Missing plugins for strategies: {_MISSING_PLUGINS}. "
        "Run `uv sync` from the repo root (one workspace covers every plugin) and re-run."
    ),
)


@_STRATEGIES_INSTALLED
def test_every_shipped_optimize_cli_contribution_resolves_and_is_callable() -> None:
    """The group is scanned by name, so a typo'd key or path drops the verb silently."""
    contributions = discover(OPTIMIZE_CLI_GROUP)

    assert "prepare-fileset" in contributions, (
        f"Expected 'prepare-fileset' in {OPTIMIZE_CLI_GROUP}, got {sorted(contributions)}. Check the "
        f'owning plugin\'s pyproject.toml [project.entry-points."{OPTIMIZE_CLI_GROUP}"] table.'
    )
    assert callable(contributions["prepare-fileset"])


@_STRATEGIES_INSTALLED
def test_every_shipped_strategy_resolves_from_real_entry_points() -> None:
    strategies = discover_strategy_jobs()

    for strategy in _REQUIRED_STRATEGY_MODULES:
        assert strategy in strategies, (
            f"Expected strategy {strategy!r} in discover_strategy_jobs(), got {sorted(strategies)}. Check the "
            'owning plugin\'s pyproject.toml [project.entry-points."nemo.jobs"] table for a stale or '
            f"misspelled 'module:ClassName' target, and that the class declares {STRATEGY_ATTR}."
        )


def test_the_router_job_submits_to_this_plugins_api_segment() -> None:
    """Resolved from the installed entry-point key, not from the module path."""
    assert (
        submit_path_for(RunStrategyJob, workspace="default")
        == "/apis/agent-optimization/v2/workspaces/default/jobs/run-strategy"
    )


def test_the_router_is_not_itself_a_strategy() -> None:
    """Otherwise `--strategy` could name the router and recurse."""
    assert not hasattr(RunStrategyJob, STRATEGY_ATTR)


@_STRATEGIES_INSTALLED
def test_the_bundled_wrapper_manifest_declares_every_optimization_job() -> None:
    """Read the wrapper manifest as data, so a typo there fails regardless of what is installed."""
    assert _WRAPPER_PYPROJECT.is_file(), f"Expected bundled wrapper manifest at {_WRAPPER_PYPROJECT}"

    with _WRAPPER_PYPROJECT.open("rb") as f:
        pyproject = tomllib.load(f)

    entry_points = pyproject["project"]["entry-points"]
    assert "nemo.optimization-strategy" not in entry_points, (
        "The retired 'nemo.optimization-strategy' entry-point group must not reappear in "
        f"{_WRAPPER_PYPROJECT} — strategies are plain nemo.jobs entries, and a known `make vendor` "
        "generator bug silently preserves this table once it exists."
    )

    jobs = entry_points["nemo.jobs"]
    assert "agents.optimize" not in jobs, (
        "`agents.optimize` is retired: the agents plugin owns no optimization job, and re-adding the "
        "key would mount a second `nemo agents optimize` command that shadows this plugin's group."
    )

    missing = sorted(({_ROUTER_JOB_KEY} | _STRATEGY_JOB_KEYS) - set(jobs))
    assert not missing, (
        f'Missing {missing} from {_WRAPPER_PYPROJECT}\'s [project.entry-points."nemo.jobs"]. '
        "A built wheel would expose no optimization strategy for them."
    )

    for key in sorted({_ROUTER_JOB_KEY} | _STRATEGY_JOB_KEYS):
        module_name, _, class_name = jobs[key].partition(":")
        assert module_name and class_name, f"Malformed entry-point target for {key!r}: {jobs[key]!r}"
        job_cls = getattr(importlib.import_module(module_name), class_name)
        assert isinstance(job_cls, type) and issubclass(job_cls, NemoJob), (
            f"{key!r} -> {jobs[key]!r} does not resolve to a NemoJob subclass."
        )

    for key in sorted(_STRATEGY_JOB_KEYS):
        module_name, _, class_name = jobs[key].partition(":")
        job_cls = getattr(importlib.import_module(module_name), class_name)
        assert declared_strategy(job_cls) is not None, (
            f"{key!r} is listed as an optimization strategy but declares no named "
            f"OptimizationStrategy in {STRATEGY_ATTR}."
        )

    agent_cli = entry_points["nemo.cli.agents"]
    assert agent_cli.get("optimize") == "nemo_agent_optimization_plugin.cli:AgentOptimizeCLI"

    optimize_cli = entry_points[OPTIMIZE_CLI_GROUP]
    assert optimize_cli.get("prepare-fileset") == ("nemo_optimization.optimize_cli:register_prepare_fileset_command")

    services = entry_points["nemo.services"]
    assert services.get("agent-optimization") == "nemo_agent_optimization_plugin.service:AgentOptimizationService"

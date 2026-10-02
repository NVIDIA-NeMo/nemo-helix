# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""This plugin plugs into ``nemo agents optimize`` exactly as the router expects.

The router finds a strategy by the ``nemo_agent_optimization_strategy`` class variable on a
``nemo.jobs`` entry, validates the forwarded fields against the strategy's own input schema,
and delegates ``compile`` / ``run``.  These tests pin each of those seams from this side, and
read the plugin's own ``pyproject.toml`` as data so a typo in the entry-point table fails here
whether or not the plugin is installed in the venv running the tests.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from nemo_agent_optimization_plugin.discovery import STRATEGY_ATTR, declared_strategy
from nemo_agent_optimization_plugin.jobs import run_strategy
from nemo_agent_optimization_plugin.jobs.run_strategy import RunStrategyJob
from nemo_agent_optimization_plugin.schemas.optimize import RunStrategySpec
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nemo_helix_plugin.jobs.execution_profiles import SubprocessJobExecutionProfile
from prompt_master_plugin.jobs import optimize as optimize_module
from prompt_master_plugin.jobs.optimize import OPTIMIZED_AGENT_FILENAME, RESULT_NAME, PromptMasterOptimizeJob

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_JOB_KEY = "agent-optimization.prompt-master"


def _pyproject() -> dict[str, Any]:
    with (PLUGIN_ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


@pytest.fixture
def installed_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the router see exactly this strategy, without touching real entry points."""
    monkeypatch.setattr(run_strategy, "discover_strategy_jobs", lambda: {"prompt-master": PromptMasterOptimizeJob})


# ---------------------------------------------------------------------------
# declaration
# ---------------------------------------------------------------------------


def test_the_job_declares_the_prompt_master_strategy() -> None:
    strategy = declared_strategy(PromptMasterOptimizeJob)

    assert strategy is not None
    assert strategy.name == "prompt-master"
    assert strategy.description, f"{STRATEGY_ATTR} should say what the strategy optimizes for list-strategies"


def test_the_entry_point_resolves_to_the_job_under_the_router_namespace() -> None:
    jobs = _pyproject()["project"]["entry-points"]["nemo.jobs"]

    assert STRATEGY_JOB_KEY in jobs, f'Expected {STRATEGY_JOB_KEY!r} in [project.entry-points."nemo.jobs"]'
    module_name, _, class_name = jobs[STRATEGY_JOB_KEY].partition(":")
    job_cls = getattr(importlib.import_module(module_name), class_name)
    assert job_cls is PromptMasterOptimizeJob
    assert issubclass(job_cls, NemoJob)


def test_the_job_name_matches_the_entry_point_key_suffix() -> None:
    """``discover_jobs`` warns on every call when NemoJob.name differs from the key's job-name part."""
    assert STRATEGY_JOB_KEY.split(".", 1)[1] == PromptMasterOptimizeJob.name


def test_the_skills_entry_point_resolves_to_the_bundled_library() -> None:
    skills = _pyproject()["project"]["entry-points"]["nemo.skills"]
    module_name, _, attr = skills["prompt-master"].partition(":")

    library = getattr(importlib.import_module(module_name), attr)()

    assert (library / "prompt-master" / "SKILL.md").is_file()


# ---------------------------------------------------------------------------
# through the router
# ---------------------------------------------------------------------------


async def _compile_through_router(spec: RunStrategySpec, *, workspace: str = "default") -> Any:
    return await RunStrategyJob.compile(
        workspace=workspace, spec=spec, entity_client=MagicMock(), job_name=None, async_sdk=MagicMock()
    )


async def test_the_router_applies_this_strategys_submit_rules(installed_strategy: None) -> None:
    """A remote submission without a staged config is refused by *this* strategy's submit schema."""
    spec = RunStrategySpec.model_validate(
        {"strategy": "prompt-master", "optimize_config": "/host/only/pm.yaml", "agent": "calculator-agent"}
    )

    with pytest.raises(HelixJobCompilationError, match="not valid for optimization strategy 'prompt-master'") as exc:
        await _compile_through_router(spec)

    assert "optimize_config_fileset is required" in str(exc.value)


async def test_the_router_requires_an_agent_for_this_strategy(installed_strategy: None) -> None:
    """The router leaves ``agent`` optional; this strategy does not, and the router honours that."""
    spec = RunStrategySpec.model_validate(
        {"strategy": "prompt-master", "optimize_config": "pm.yaml", "optimize_config_fileset": "default/pm-bundle"}
    )

    with pytest.raises(HelixJobCompilationError, match="not valid for optimization strategy 'prompt-master'") as exc:
        await _compile_through_router(spec)

    assert "agent" in str(exc.value)


async def test_the_router_dispatches_compile_to_this_job(installed_strategy: None) -> None:
    spec = RunStrategySpec.model_validate(
        {
            "strategy": "prompt-master",
            "optimize_config": "pm.yaml",
            "optimize_config_fileset": "default/pm-bundle",
            "agent": "calculator-agent",
            "output": "default/pm-results",
        }
    )

    async def _get_execution_profiles() -> Any:
        return SimpleNamespace(data=lambda: [SubprocessJobExecutionProfile(profile="default")])

    jobs_client = MagicMock()
    jobs_client.get_execution_profiles = _get_execution_profiles
    with patch.object(optimize_module, "client_from_platform", return_value=jobs_client):
        compiled = await _compile_through_router(spec, workspace="staging")

    (step,) = compiled.steps
    assert step.name == "prompt-master"
    assert step.executor.command == ["python", "-m", "prompt_master_plugin.tasks.optimize"]
    # The router forwards its fields minus ``strategy``; the workspace is the submission's.
    assert step.config == {
        "optimize_config": "pm.yaml",
        "optimize_config_fileset": "default/pm-bundle",
        "agent": "calculator-agent",
        "output": "default/pm-results",
        "workspace": "staging",
    }


def test_the_router_runs_this_job_in_process_with_the_platform_client(
    installed_strategy: None,
    ctx: JobContext,
    config_path: Path,
    tmp_path: Path,
    fake_prompt_master: list[Any],
    stored_agent: Any,
    source_agent: dict[str, Any],
) -> None:
    """The local path: the router's ``run`` forwards ``sdk`` because this job's ``run`` accepts it."""
    config = {"strategy": "prompt-master", "optimize_config": str(config_path), "agent": "calculator-agent"}

    with stored_agent(source_agent):
        result = RunStrategyJob().run(config, ctx=ctx, sdk=MagicMock())

    assert result["status"] == "completed"
    assert result["agent"] == "default/calculator-agent"
    assert (tmp_path / "job-results" / RESULT_NAME / OPTIMIZED_AGENT_FILENAME).is_file()

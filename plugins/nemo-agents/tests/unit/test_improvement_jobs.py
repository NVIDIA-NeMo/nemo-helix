# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the new agent-improvement NemoJobs are discoverable + well-formed."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError


def test_jobs_discovered_via_entry_points() -> None:
    from nemo_helix_plugin.discovery import discover_jobs

    jobs = discover_jobs()
    assert "agents.evaluate-suite" in jobs
    assert "agents.analyze" in jobs
    assert "agents.optimize-skills" in jobs
    # Optimization is not an agents job: nemo-agent-optimization-plugin owns the
    # `nemo agents optimize` group and registers its own job and routes.
    assert "agents.optimize" not in jobs


def test_evaluate_suite_job_metadata() -> None:
    from nemo_agents_plugin.jobs.evaluate_suite import EvaluateSuiteJob

    assert EvaluateSuiteJob.name == "evaluate-suite"
    assert EvaluateSuiteJob.container == "cpu-tasks"


def test_analyze_job_metadata() -> None:
    from nemo_agents_plugin.jobs.analyze_batch import AnalyzeBatchJob

    assert AnalyzeBatchJob.name == "analyze"
    assert AnalyzeBatchJob.container == "cpu-tasks"


def test_optimize_skills_job_metadata() -> None:
    from nemo_agents_plugin.jobs.optimize_skills import OptimizeSkillsJob

    assert OptimizeSkillsJob.name == "optimize-skills"
    assert OptimizeSkillsJob.container == "cpu-tasks"


def test_evaluate_suite_config_validation() -> None:
    from nemo_agents_plugin.jobs.evaluate_suite import EvaluateSuiteConfig

    cfg = EvaluateSuiteConfig.model_validate({"evals": "./my-evals"})
    assert cfg.evals == "./my-evals"
    assert cfg.runner == "auto"
    assert cfg.prefer == "nat"
    assert cfg.concurrency == 4


def test_evaluate_suite_documented_submit_shape_validates() -> None:
    from nemo_agents_plugin.jobs.evaluate_suite import EvaluateSuiteSubmitConfig

    cfg = EvaluateSuiteSubmitConfig.model_validate(
        {
            "evals": "/path/to/my-agent/tests/agentic",
            "agent": "/path/to/my-agent",
            "output": "/path/to/my-agent/runs/batch-smoke",
            "anthropic_api_key_secret": "anthropic-api-key",
            "anthropic_base_url": "https://inference-api.nvidia.com",
        }
    )

    assert cfg.evals == "/path/to/my-agent/tests/agentic"
    assert cfg.agent == "/path/to/my-agent"
    assert cfg.output == "/path/to/my-agent/runs/batch-smoke"
    assert cfg.anthropic_api_key_secret == "anthropic-api-key"
    assert cfg.anthropic_base_url == "https://inference-api.nvidia.com"


def test_optimize_skills_config_validation() -> None:
    from nemo_agents_plugin.jobs.optimize_skills import OptimizeSkillsConfig

    cfg = OptimizeSkillsConfig.model_validate({"evals": "./e", "agent": "./a"})
    assert cfg.skills_path == ".agents/skills"
    assert cfg.iterations == 3
    assert cfg.repeats == 1
    assert cfg.open_pr is False


# ---------------------------------------------------------------------------
# compile() — platform-job dispatch shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_suite_compile_produces_single_subprocess_step() -> None:
    from nemo_agents_plugin.jobs.evaluate_suite import EvaluateSuiteConfig, EvaluateSuiteJob
    from nemo_helix_plugin.jobs.constants import (
        DEFAULT_JOB_STORAGE_PATH,
        PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
    )

    spec = EvaluateSuiteConfig(evals="/abs/evals", agent="/abs/agent", output="/abs/out")
    platform_spec = await EvaluateSuiteJob.compile(
        workspace="default",
        spec=spec,
        entity_client=MagicMock(),
        job_name=None,
        async_sdk=MagicMock(),
    )
    steps = list(platform_spec["steps"])
    assert len(steps) == 1
    step = steps[0]
    assert step["name"] == "evaluate-suite"
    assert step["executor"]["provider"] == "subprocess"
    assert step["executor"]["command"] == ["python", "-m", "nemo_agents_plugin.tasks.evaluate_suite"]
    assert step["config"]["evals"] == "/abs/evals"
    assert step["config"]["agent"] == "/abs/agent"

    env = {e["name"]: e.get("value") for e in step["environment"]}
    assert env[PERSISTENT_JOB_STORAGE_PATH_ENVVAR] == DEFAULT_JOB_STORAGE_PATH


@pytest.mark.asyncio
async def test_evaluate_suite_compile_url_workspace_overrides_spec_workspace() -> None:
    from nemo_agents_plugin.jobs.evaluate_suite import EvaluateSuiteConfig, EvaluateSuiteJob

    spec = EvaluateSuiteConfig(evals="/abs/evals", agent="/abs/agent", output="/abs/out")
    platform_spec = await EvaluateSuiteJob.compile(
        workspace="staging",
        spec=spec,
        entity_client=MagicMock(),
        job_name=None,
        async_sdk=MagicMock(),
    )
    config = next(iter(platform_spec["steps"]))["config"]
    assert config["workspace"] == "staging"


@pytest.mark.asyncio
async def test_evaluate_suite_compile_rejects_relative_paths() -> None:
    from nemo_agents_plugin.jobs.evaluate_suite import EvaluateSuiteConfig, EvaluateSuiteJob

    spec = EvaluateSuiteConfig(evals="./my-evals", agent="/abs/agent", output="/abs/out")
    with pytest.raises(HelixJobCompilationError, match="'evals' must be an absolute path"):
        await EvaluateSuiteJob.compile(
            workspace="default",
            spec=spec,
            entity_client=MagicMock(),
            job_name=None,
            async_sdk=MagicMock(),
        )


@pytest.mark.asyncio
async def test_evaluate_suite_compile_rejects_none_agent() -> None:
    from nemo_agents_plugin.jobs.evaluate_suite import EvaluateSuiteConfig, EvaluateSuiteJob

    spec = EvaluateSuiteConfig(evals="/abs/evals", output="/abs/out")  # agent defaults to None
    with pytest.raises(HelixJobCompilationError, match="'agent' is required"):
        await EvaluateSuiteJob.compile(
            workspace="default",
            spec=spec,
            entity_client=MagicMock(),
            job_name=None,
            async_sdk=MagicMock(),
        )


@pytest.mark.asyncio
async def test_evaluate_suite_compile_rejects_none_output() -> None:
    from nemo_agents_plugin.jobs.evaluate_suite import EvaluateSuiteConfig, EvaluateSuiteJob

    spec = EvaluateSuiteConfig(evals="/abs/evals", agent="/abs/agent")  # output defaults to None
    with pytest.raises(HelixJobCompilationError, match="'output' is required"):
        await EvaluateSuiteJob.compile(
            workspace="default",
            spec=spec,
            entity_client=MagicMock(),
            job_name=None,
            async_sdk=MagicMock(),
        )


@pytest.mark.asyncio
async def test_evaluate_suite_compile_injects_anthropic_secret_when_set() -> None:
    from nemo_agents_plugin.jobs.evaluate_suite import EvaluateSuiteConfig, EvaluateSuiteJob

    spec = EvaluateSuiteConfig(
        evals="/abs/evals",
        agent="/abs/agent",
        output="/abs/out",
        anthropic_api_key_secret="anthropic-api-key",
        anthropic_base_url="https://inference-api.nvidia.com",
    )
    platform_spec = await EvaluateSuiteJob.compile(
        workspace="default",
        spec=spec,
        entity_client=MagicMock(),
        job_name=None,
        async_sdk=MagicMock(),
    )
    step = next(iter(platform_spec["steps"]))
    env = {e["name"]: e for e in step["environment"]}
    assert env["ANTHROPIC_API_KEY"]["from_secret"]["name"] == "anthropic-api-key"
    assert env["ANTHROPIC_BASE_URL"]["value"] == "https://inference-api.nvidia.com"


@pytest.mark.asyncio
async def test_optimize_skills_compile_produces_single_subprocess_step() -> None:
    from nemo_agents_plugin.jobs.optimize_skills import OptimizeSkillsConfig, OptimizeSkillsJob
    from nemo_helix_plugin.jobs.constants import (
        DEFAULT_JOB_STORAGE_PATH,
        PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
    )

    spec = OptimizeSkillsConfig(evals="/abs/evals", agent="/abs/agent")
    platform_spec = await OptimizeSkillsJob.compile(
        workspace="default",
        spec=spec,
        entity_client=MagicMock(),
        job_name=None,
        async_sdk=MagicMock(),
    )
    steps = list(platform_spec["steps"])
    assert len(steps) == 1
    step = steps[0]
    assert step["name"] == "optimize-skills"
    executor = step["executor"]
    assert executor.get("provider") == "subprocess"
    assert executor.get("command") == ["python", "-m", "nemo_agents_plugin.tasks.optimize_skills"]

    env = {e["name"]: e for e in step["environment"]}
    assert env[PERSISTENT_JOB_STORAGE_PATH_ENVVAR]["value"] == DEFAULT_JOB_STORAGE_PATH
    # When ``anthropic_api_key_secret`` is unset, no ANTHROPIC_API_KEY env is declared.
    assert "ANTHROPIC_API_KEY" not in env


@pytest.mark.asyncio
async def test_optimize_skills_compile_injects_anthropic_secret_when_set() -> None:
    from nemo_agents_plugin.jobs.optimize_skills import OptimizeSkillsConfig, OptimizeSkillsJob

    spec = OptimizeSkillsConfig(
        evals="/abs/evals",
        agent="/abs/agent",
        anthropic_api_key_secret="anthropic-api-key",
        anthropic_base_url="https://inference-api.nvidia.com",
    )
    platform_spec = await OptimizeSkillsJob.compile(
        workspace="default",
        spec=spec,
        entity_client=MagicMock(),
        job_name=None,
        async_sdk=MagicMock(),
    )
    step = next(iter(platform_spec["steps"]))
    env = {e["name"]: e for e in step["environment"]}
    assert env["ANTHROPIC_API_KEY"]["from_secret"]["name"] == "anthropic-api-key"
    assert env["ANTHROPIC_BASE_URL"]["value"] == "https://inference-api.nvidia.com"


@pytest.mark.asyncio
async def test_optimize_skills_compile_rejects_relative_paths() -> None:
    from nemo_agents_plugin.jobs.optimize_skills import OptimizeSkillsConfig, OptimizeSkillsJob

    spec = OptimizeSkillsConfig(evals="/abs/evals", agent="./my-agent")
    with pytest.raises(HelixJobCompilationError, match="'agent' must be an absolute path"):
        await OptimizeSkillsJob.compile(
            workspace="default",
            spec=spec,
            entity_client=MagicMock(),
            job_name=None,
            async_sdk=MagicMock(),
        )


@pytest.mark.asyncio
async def test_optimize_skills_compile_rejects_analyze_only_without_initial_batch() -> None:
    from nemo_agents_plugin.jobs.optimize_skills import OptimizeSkillsConfig, OptimizeSkillsJob

    spec = OptimizeSkillsConfig(evals="/abs/evals", agent="/abs/agent", analyze_only=True)
    with pytest.raises(HelixJobCompilationError, match="'analyze_only' requires 'initial_batch'"):
        await OptimizeSkillsJob.compile(
            workspace="default",
            spec=spec,
            entity_client=MagicMock(),
            job_name=None,
            async_sdk=MagicMock(),
        )


# ---------------------------------------------------------------------------
# AnalyzeBatchJob — newly added compile() paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_compile_produces_single_subprocess_step() -> None:
    from nemo_agents_plugin.jobs.analyze_batch import AnalyzeBatchConfig, AnalyzeBatchJob

    spec = AnalyzeBatchConfig(batch="/abs/batch", mechanical_only=True)
    platform_spec = await AnalyzeBatchJob.compile(
        workspace="default",
        spec=spec,
        entity_client=MagicMock(),
        job_name=None,
        async_sdk=MagicMock(),
    )
    step = next(iter(platform_spec["steps"]))
    assert step["name"] == "analyze"
    executor = step["executor"]
    assert executor.get("provider") == "subprocess"
    assert executor.get("command") == ["python", "-m", "nemo_agents_plugin.tasks.analyze"]


@pytest.mark.asyncio
async def test_analyze_compile_injects_anthropic_env_when_set() -> None:
    from nemo_agents_plugin.jobs.analyze_batch import AnalyzeBatchConfig, AnalyzeBatchJob

    spec = AnalyzeBatchConfig(
        batch="/abs/batch",
        anthropic_api_key_secret="anthropic-api-key",
        anthropic_base_url="https://inference-api.nvidia.com",
    )
    platform_spec = await AnalyzeBatchJob.compile(
        workspace="default",
        spec=spec,
        entity_client=MagicMock(),
        job_name=None,
        async_sdk=MagicMock(),
    )
    step = next(iter(platform_spec["steps"]))
    env = {e["name"]: e for e in step["environment"]}
    assert env["ANTHROPIC_API_KEY"]["from_secret"]["name"] == "anthropic-api-key"
    assert env["ANTHROPIC_BASE_URL"]["value"] == "https://inference-api.nvidia.com"


@pytest.mark.asyncio
async def test_analyze_compile_requires_anthropic_secret_unless_mechanical_only() -> None:
    from nemo_agents_plugin.jobs.analyze_batch import AnalyzeBatchConfig, AnalyzeBatchJob

    spec = AnalyzeBatchConfig(batch="/abs/batch")  # mechanical_only defaults False, secret unset
    with pytest.raises(HelixJobCompilationError, match="anthropic_api_key_secret"):
        await AnalyzeBatchJob.compile(
            workspace="default",
            spec=spec,
            entity_client=MagicMock(),
            job_name=None,
            async_sdk=MagicMock(),
        )


@pytest.mark.asyncio
async def test_analyze_compile_rejects_relative_batch_path() -> None:
    from nemo_agents_plugin.jobs.analyze_batch import AnalyzeBatchConfig, AnalyzeBatchJob

    spec = AnalyzeBatchConfig(batch="./my-batch", mechanical_only=True)
    with pytest.raises(HelixJobCompilationError, match="'batch' must be an absolute path"):
        await AnalyzeBatchJob.compile(
            workspace="default",
            spec=spec,
            entity_client=MagicMock(),
            job_name=None,
            async_sdk=MagicMock(),
        )

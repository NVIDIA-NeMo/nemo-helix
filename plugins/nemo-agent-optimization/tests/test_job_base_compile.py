# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any, ClassVar

import pytest
from nemo_agent_optimization_plugin import job_base
from nemo_agent_optimization_plugin.job_base import AgentOptimizeJob
from nemo_platform_plugin.jobs.exceptions import (
    PlatformJobCompilationError,
    PlatformJobDependencyUnavailableError,
)


class _Job(AgentOptimizeJob):
    name: ClassVar[str] = "agent_optimize"
    strategy: ClassVar[str] = "fake"
    task_module: ClassVar[str] = "fake_plugin.tasks.agent_optimize"

    def optimize(self, **kwargs: Any) -> dict[str, Any]:
        return {}


def _spec() -> Any:
    from nemo_agent_optimization_plugin.schemas.optimize import AgentOptimizeSpec

    return AgentOptimizeSpec.model_validate(
        {
            "agent": "my-ws/my-agent",
            "config_fileset": "my-ws/bundle",
            "config": "configs/optimize.yaml",
            "output_agent": "my-agent-opt",
            "workspace": "my-ws",
        }
    )


class _Profile:
    def __init__(self, provider: str, profile: str) -> None:
        self.provider = provider
        self.profile = profile


def _compile(monkeypatch: pytest.MonkeyPatch, profiles: list[Any]) -> Any:
    # ``_fetch_execution_profiles`` is genuinely async in job_base (it awaits
    # AsyncJobsClient.get_execution_profiles(), which returns an awaitable), and
    # ``_resolve_executor`` awaits it. The replacement here has to be an async callable
    # too so that await keeps working under the patch — a plain sync lambda would make
    # ``await _fetch_execution_profiles(...)`` fail with "object list can't be used in
    # 'await' expression".
    async def fake_fetch_execution_profiles(async_sdk: object) -> list[Any]:
        return profiles

    monkeypatch.setattr(job_base, "_fetch_execution_profiles", fake_fetch_execution_profiles)
    return asyncio.run(
        _Job.compile(
            workspace="my-ws",
            spec=_spec(),
            entity_client=None,
            job_name=None,
            async_sdk=object(),
        )
    )


def test_compile_produces_one_step_carrying_the_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = _compile(monkeypatch, [_Profile("cpu", "default")])

    assert len(compiled.steps) == 1
    assert compiled.steps[0].config["output_agent"] == "my-agent-opt"
    assert compiled.steps[0].config["workspace"] == "my-ws"


def test_the_cpu_executor_runs_the_subclass_task_module(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = _compile(monkeypatch, [_Profile("cpu", "default")])
    executor = compiled.steps[0].executor

    assert executor.provider == "cpu"
    assert executor.container.command == ["fake_plugin.tasks.agent_optimize"]
    assert executor.container.entrypoint == ["python", "-m"]


def test_a_subprocess_profile_is_preferred_when_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_platform_plugin.jobs.execution_profiles import SubprocessJobExecutionProfile

    subprocess_profile = SubprocessJobExecutionProfile.model_construct(provider="subprocess", profile="default")
    compiled = _compile(monkeypatch, [subprocess_profile, _Profile("cpu", "default")])
    executor = compiled.steps[0].executor

    assert executor.provider == "subprocess"
    assert executor.command == ["python", "-m", "fake_plugin.tasks.agent_optimize"]


def test_no_matching_profile_is_a_compilation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(PlatformJobCompilationError, match="nowhere to run"):
        _compile(monkeypatch, [_Profile("gpu", "other")])


def test_a_missing_async_sdk_is_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(PlatformJobDependencyUnavailableError, match="temporarily"):
        asyncio.run(
            _Job.compile(
                workspace="my-ws",
                spec=_spec(),
                entity_client=None,
                job_name=None,
                async_sdk=None,
            )
        )


def test_a_subclass_without_a_task_module_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoModule(AgentOptimizeJob):
        name: ClassVar[str] = "agent_optimize"
        strategy: ClassVar[str] = "nomodule"

        def optimize(self, **kwargs: Any) -> dict[str, Any]:
            return {}

    async def fake_fetch_execution_profiles(async_sdk: object) -> list[Any]:
        return [_Profile("cpu", "default")]

    monkeypatch.setattr(job_base, "_fetch_execution_profiles", fake_fetch_execution_profiles)
    with pytest.raises(PlatformJobCompilationError, match="task_module"):
        asyncio.run(
            _NoModule.compile(
                workspace="my-ws",
                spec=_spec(),
                entity_client=None,
                job_name=None,
                async_sdk=object(),
            )
        )

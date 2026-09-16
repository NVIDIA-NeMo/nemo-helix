# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any

import pytest
from nemo_agent_optimization_plugin.job_base import AgentOptimizeJob
from nemo_agent_optimization_plugin.jobs import optimize as router
from nemo_agent_optimization_plugin.jobs.optimize import OptimizeJob
from nemo_platform_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    PlatformJobSpec,
    PlatformJobStep,
)
from nemo_platform_plugin.run_dependencies import LocalRunError

_FAKE_EXECUTOR = CPUExecutionProviderSpec(provider="cpu", profile="default", container=ContainerSpec())


class _Target(AgentOptimizeJob):
    name = "agent_optimize"
    strategy = "fake"
    compiled: list[dict[str, Any]] = []

    def optimize(self, **kwargs):
        return {}

    @classmethod
    async def compile(cls, *, workspace, spec, **kwargs):
        cls.compiled.append({"workspace": workspace, "spec": spec})
        return PlatformJobSpec(steps=[PlatformJobStep(name="fake-step", executor=_FAKE_EXECUTOR, config={})])


def _spec(**overrides: Any) -> dict[str, Any]:
    return {
        "strategy": "fake",
        "agent": "my-ws/my-agent",
        "optimize_config_fileset": "my-ws/bundle",
        "optimize_config": "configs/optimize.yaml",
        "output_agent": "my-agent-opt",
        "workspace": "my-ws",
        **overrides,
    }


@pytest.fixture(autouse=True)
def installed(monkeypatch: pytest.MonkeyPatch) -> None:
    _Target.compiled = []
    monkeypatch.setattr(router, "discover_agent_optimize_jobs", lambda: {"fake": _Target})


def test_compile_delegates_to_the_strategy_job_and_splices_its_steps() -> None:
    spec = router.OptimizeSpec.model_validate(_spec())
    compiled = asyncio.run(
        OptimizeJob.compile(workspace="my-ws", spec=spec, entity_client=None, job_name=None, async_sdk=None)
    )

    assert [step.name for step in compiled.steps] == ["fake-step"]
    assert _Target.compiled[0]["workspace"] == "my-ws"


def test_compile_strips_strategy_from_the_child_spec() -> None:
    spec = router.OptimizeSpec.model_validate(_spec())
    asyncio.run(OptimizeJob.compile(workspace="my-ws", spec=spec, entity_client=None, job_name=None, async_sdk=None))

    child = _Target.compiled[0]["spec"]
    assert not hasattr(child, "strategy")
    assert child.output_agent == "my-agent-opt"


def test_an_unknown_strategy_is_rejected_with_the_installed_list() -> None:
    spec = router.OptimizeSpec.model_validate(_spec(strategy="nope"))
    with pytest.raises(LocalRunError, match=r"not installed\. Available strategies: \['fake'\]"):
        asyncio.run(
            OptimizeJob.compile(workspace="my-ws", spec=spec, entity_client=None, job_name=None, async_sdk=None)
        )


def test_run_delegates_to_run_local_with_the_child_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class _FakeScheduler:
        def run_local(self, job_cls, spec, *, workspace, sdk=None):
            calls.append({"job_cls": job_cls, "spec": spec, "workspace": workspace})
            return {"agent": "my-ws/my-agent-opt"}

    monkeypatch.setattr(router, "NemoJobScheduler", _FakeScheduler)
    result = OptimizeJob().run(_spec(), ctx=object(), sdk=object())  # ty: ignore[invalid-argument-type]

    assert result == {"agent": "my-ws/my-agent-opt"}
    assert calls[0]["job_cls"] is _Target
    assert calls[0]["spec"]["output_agent"] == "my-agent-opt"
    assert "strategy" not in calls[0]["spec"]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``switchyard`` strategy job: its compiled step and its run."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from nemo_agent_optimization_plugin.discovery import declared_strategy
from nemo_helix_plugin.errors import LocalRunError
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nemo_helix_plugin.jobs.execution_profiles import (
    DockerJobExecutionProfile,
    DockerJobExecutionProfileConfig,
    SubprocessJobExecutionProfile,
)
from nemo_helix_plugin.virtual_models.types import VirtualModel, VirtualModelInferenceConfig
from nemo_switchyard.jobs import optimize as optimize_module
from nemo_switchyard.jobs.optimize import INDEX_FILENAME, RESULT_NAME, TASK_MODULE, SwitchyardOptimizeJob
from nemo_switchyard.schemas.optimize import SwitchyardOptimizeSpec

SUBPROCESS_PROFILE = SubprocessJobExecutionProfile(profile="default")
CPU_PROFILE = DockerJobExecutionProfile(provider="cpu", profile="default", config=DockerJobExecutionProfileConfig())
SPEC: dict[str, Any] = {"agent": "calc", "models": ["a", "b"], "routing_strategies": ["random_routing", "stage_router"]}
SOURCE_AGENT: dict[str, Any] = {
    "config_format": "nemo-agents-spec-v1",
    "name": "calc",
    "harnesses": {"deepagents": {"kind": "deepagents"}},
    "models": {"default": {"provider": "nvidia", "model": "m", "api_key_env": "NVIDIA_API_KEY"}},
}


@pytest.fixture
def ctx(tmp_path: Path) -> JobContext:
    ephemeral = tmp_path / "ephemeral"
    ephemeral.mkdir()
    return JobContext(
        workspace="default",
        storage=StoragePaths(ephemeral=ephemeral),
        results=LocalJobResults(root=tmp_path / "job-results"),
    )


@contextlib.contextmanager
def platform(
    *execution_profiles: Any, agent: dict[str, Any] | None = None, existing: VirtualModel | None = None
) -> Iterator[MagicMock]:
    async def _get_execution_profiles() -> Any:
        return SimpleNamespace(data=lambda: list(execution_profiles))

    def _create_virtual_model(**call: Any) -> Any:
        body = call["body"]
        created = existing or VirtualModel(
            name=body.name, workspace=call["workspace"], models=body.models, request_middleware=body.request_middleware
        )
        return SimpleNamespace(data=lambda: created)

    client = MagicMock()
    client.get_execution_profiles = _get_execution_profiles
    client.get_agent.return_value.data.return_value = SimpleNamespace(config=agent)
    client.create_virtual_model.side_effect = _create_virtual_model
    with patch.object(optimize_module, "client_from_platform", return_value=client):
        yield client


async def compile_spec(profile: str | None = None) -> Any:
    return await SwitchyardOptimizeJob.compile(
        workspace="staging",
        spec=SwitchyardOptimizeSpec.model_validate(SPEC),
        entity_client=MagicMock(),
        job_name=None,
        async_sdk=MagicMock(),
        profile=profile,
    )


def test_declares_the_strategy() -> None:
    strategy = declared_strategy(SwitchyardOptimizeJob)
    assert strategy is not None
    assert strategy.name == "switchyard"


async def test_compile_prefers_the_subprocess_executor_and_stamps_the_workspace() -> None:
    with platform(SUBPROCESS_PROFILE, CPU_PROFILE):
        (step,) = (await compile_spec()).steps

    assert step.name == "switchyard"
    assert step.executor.provider == "subprocess"
    assert step.executor.command == ["python", "-m", TASK_MODULE]
    assert step.config["workspace"] == "staging"
    assert step.config["models"] == ["a", "b"]


async def test_compile_falls_back_to_the_cpu_container() -> None:
    with (
        platform(CPU_PROFILE),
        patch.object(optimize_module, "get_qualified_image", return_value="reg.example/nhx-tasks:test"),
    ):
        (step,) = (await compile_spec(profile="default")).steps

    assert step.executor.provider == "cpu"
    assert step.executor.container.image == "reg.example/nhx-tasks:test"
    assert step.executor.container.command == [TASK_MODULE]


async def test_compile_fails_without_a_matching_profile() -> None:
    with platform(DockerJobExecutionProfile(provider="gpu", profile="a100", config=DockerJobExecutionProfileConfig())):
        with pytest.raises(HelixJobCompilationError, match=r"Available profiles: \['gpu/a100'\]"):
            await compile_spec()


def run_job(ctx: JobContext) -> dict[str, Any]:
    return SwitchyardOptimizeJob().run({**SPEC, "agent": "team/calc", "workspace": "default"}, ctx=ctx, sdk=MagicMock())


def test_run_creates_one_virtual_model_per_combination(ctx: JobContext) -> None:
    with platform(agent=SOURCE_AGENT) as client:
        result = run_job(ctx)

    client.get_agent.assert_called_once_with(name="calc", workspace="team")
    calls = client.create_virtual_model.call_args_list
    assert all(call.kwargs["workspace"] == "default" and call.kwargs["exist_ok"] for call in calls)
    bodies = [call.kwargs["body"] for call in calls]
    assert [body.name for body in bodies] == ["calc-random-routing-1", "calc-stage-router-1"]
    assert [m.model for m in bodies[0].models] == ["default/a", "default/b"]
    (call,) = bodies[1].request_middleware
    assert call.name == "nemo-switchyard"
    assert call.config_type == "stage_router"
    assert call.config["models"] == {"capable": ["default/a"], "efficient": ["default/b"]}
    assert result["virtual_models"] == ["calc-random-routing-1", "calc-stage-router-1"]


def test_run_writes_one_agent_config_each_and_an_index(ctx: JobContext, tmp_path: Path) -> None:
    with platform(agent=SOURCE_AGENT):
        result = run_job(ctx)

    assert result["status"] == "completed"
    assert result["agent"] == "team/calc"
    assert result["result"]["name"] == RESULT_NAME
    saved = tmp_path / "job-results" / RESULT_NAME
    rewritten = yaml.safe_load((saved / "agent-calc-random-routing-1.yaml").read_text(encoding="utf-8"))
    assert rewritten["models"]["default"]["model"] == "default/calc-random-routing-1"
    index = json.loads((saved / INDEX_FILENAME).read_text(encoding="utf-8"))
    assert index["agent"] == "team/calc"
    assert [c["agent_config"] for c in index["combinations"]] == [
        "agent-calc-random-routing-1.yaml",
        "agent-calc-stage-router-1.yaml",
    ]
    assert index["combinations"][0]["config"]["strong_probability"] == 0.5


@pytest.mark.parametrize("config", [{}, {"schema_version": "fabric.agent/v1alpha1"}])
def test_run_refuses_an_agent_that_is_not_spec_v1(ctx: JobContext, config: dict[str, Any]) -> None:
    with platform(agent=config), pytest.raises(LocalRunError, match="nemo-agents-spec-v1"):
        SwitchyardOptimizeJob().run({**SPEC, "workspace": "default"}, ctx=ctx, sdk=MagicMock())


def test_run_refuses_an_existing_virtual_model_that_routes_differently(ctx: JobContext) -> None:
    stale = VirtualModel(
        name="calc-random-routing-1",
        workspace="default",
        models=[VirtualModelInferenceConfig(model="default/a"), VirtualModelInferenceConfig(model="default/c")],
    )
    with platform(agent=SOURCE_AGENT, existing=stale), pytest.raises(LocalRunError, match="routes differently"):
        run_job(ctx)


def test_run_creates_no_virtual_model_when_the_agent_has_nothing_to_route(ctx: JobContext) -> None:
    with (
        platform(agent={**SOURCE_AGENT, "models": {}}) as client,
        pytest.raises(LocalRunError, match="nothing to rewrite"),
    ):
        run_job(ctx)

    client.create_virtual_model.assert_not_called()

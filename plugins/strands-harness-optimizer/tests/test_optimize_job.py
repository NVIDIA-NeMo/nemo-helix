# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``strands-harness-optimizer`` strategy job: its spec, its compiled step, and its run."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import httpx
import pytest
import yaml
from nemo_helix_plugin.client.errors import NemoTransportError, NotFoundError
from nemo_helix_plugin.errors import LocalRunError
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError, HelixJobDependencyUnavailableError
from nemo_helix_plugin.jobs.execution_profiles import (
    DockerJobExecutionProfile,
    DockerJobExecutionProfileConfig,
    SubprocessJobExecutionProfile,
)
from pydantic import ValidationError
from strands_harness_optimizer_plugin.config import StrandsHarnessConfig
from strands_harness_optimizer_plugin.jobs import optimize as optimize_module
from strands_harness_optimizer_plugin.jobs.optimize import (
    OPTIMIZED_AGENT_FILENAME,
    RESULT_NAME,
    SUMMARY_FILENAME,
    TASK_MODULE,
    StrandsHarnessOptimizeJob,
)
from strands_harness_optimizer_plugin.schemas.optimize import StrandsHarnessOptimizeSpec
from strands_harness_optimizer_plugin.strands_bridge import Outcome

SUBPROCESS_PROFILE = SubprocessJobExecutionProfile(profile="default")
CPU_PROFILE = DockerJobExecutionProfile(provider="cpu", profile="default", config=DockerJobExecutionProfileConfig())
GPU_PROFILE = DockerJobExecutionProfile(provider="gpu", profile="a100", config=DockerJobExecutionProfileConfig())
STAGED: dict[str, Any] = {
    "agent": "calculator-agent",
    "optimize_config": "strands-harness-optimizer.yaml",
    "optimize_config_fileset": "default/strands-optimizer-bundle",
}
ROWS = [{"input": f"What is {i} + {i}?", "expected_output": str(2 * i)} for i in range(4)]
UNREACHABLE = NemoTransportError(httpx.ConnectError("refused", request=httpx.Request("GET", "http://x")))


async def compile_with(*profiles: Any, error: Exception | None = None, profile: str | None = None) -> Any:
    client = MagicMock()
    client.get_execution_profiles = AsyncMock(return_value=SimpleNamespace(data=lambda: profiles), side_effect=error)
    with patch.object(optimize_module, "client_from_platform", return_value=client):
        return await StrandsHarnessOptimizeJob.compile(
            workspace="staging",
            spec=StrandsHarnessOptimizeSpec.model_validate(STAGED),
            entity_client=None,
            job_name=None,
            async_sdk=MagicMock(),
            profile=profile,
        )


def run_job(ctx: JobContext, **overrides: Any) -> dict[str, Any]:
    return StrandsHarnessOptimizeJob().run({**STAGED, "workspace": "default", **overrides}, ctx=ctx, sdk=MagicMock())


@pytest.fixture
def bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    def _stage(**settings: Any) -> None:
        (tmp_path / "dataset.jsonl").write_text("".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8")
        yaml_path = tmp_path / "strands-harness-optimizer.yaml"
        yaml_path.write_text(yaml.safe_dump({"dataset": "dataset.jsonl", **settings}), encoding="utf-8")

        @contextlib.contextmanager
        def resolve(*args: Any, **kwargs: Any) -> Generator[Path]:
            yield yaml_path

        monkeypatch.setattr(optimize_module, "resolve_staged_config", resolve)

    return _stage


@pytest.fixture
def agents(monkeypatch: pytest.MonkeyPatch, source_agent: dict[str, Any]) -> MagicMock:
    agents = MagicMock()
    agents.get_agent.return_value.data.return_value.config = source_agent
    monkeypatch.setattr(optimize_module, "client_from_platform", lambda sdk, cls: agents)
    return agents


@pytest.fixture(autouse=True)
def optimize(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    outcome = Outcome(original_prompt="old prompt", optimized_prompt="new prompt", epoch_stats=[{"epoch": 1}])
    optimize = MagicMock(return_value=outcome)
    monkeypatch.setattr(optimize_module, "optimize_system_prompt", optimize)
    monkeypatch.setattr(optimize_module, "platform_auth_proxy", lambda: contextlib.nullcontext("http://127.0.0.1:9"))
    return optimize


@pytest.mark.parametrize(
    "override",
    [{"agent": "https://example.com/agent"}, {"optimize_config_fileset": "too/many/parts"}, {"output": "default/out"}],
)
def test_spec_rejects(override: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match=next(iter(override))):
        StrandsHarnessOptimizeSpec.model_validate({**STAGED, **override})


async def test_compile_prefers_subprocess() -> None:
    (step,) = (await compile_with(SUBPROCESS_PROFILE, CPU_PROFILE)).steps

    assert step.name == "strands-harness-optimizer"
    assert step.executor.provider == "subprocess"
    assert step.executor.command == ["python", "-m", TASK_MODULE]
    assert step.config == {**STAGED, "workspace": "staging"}


async def test_compile_falls_back_to_cpu() -> None:
    with patch.object(optimize_module, "get_qualified_image", return_value="reg/nhx-tasks:t"):
        (step,) = (await compile_with(CPU_PROFILE, profile="default")).steps

    assert step.executor.provider == "cpu"
    assert step.executor.container.image == "reg/nhx-tasks:t"
    assert step.executor.container.command == [TASK_MODULE]


@pytest.mark.parametrize(
    ("profiles", "error", "raised", "message"),
    [
        ([GPU_PROFILE], None, HelixJobCompilationError, r"Available profiles: \['gpu/a100'\]"),
        ([], UNREACHABLE, HelixJobDependencyUnavailableError, "temporarily unavailable"),
    ],
)
async def test_compile_errors(
    profiles: list[Any], error: Exception | None, raised: type[Exception], message: str
) -> None:
    with pytest.raises(raised, match=message):
        await compile_with(*profiles, error=error)


def test_run(
    ctx: JobContext,
    tmp_path: Path,
    bundle: Callable[..., None],
    agents: MagicMock,
    optimize: MagicMock,
    source_agent: dict[str, Any],
) -> None:
    bundle(epochs=3, max_samples=2)

    result = run_job(ctx, agent="team/calculator-agent", workspace="staging")

    assert result == {
        "status": "completed",
        "strategy": "strands-harness-optimizer",
        "agent": "team/calculator-agent",
        "optimized_prompt": "new prompt",
        "result": ANY,
    }
    assert result["result"]["name"] == RESULT_NAME
    agents.get_agent.assert_called_once_with(name="calculator-agent", workspace="team")
    optimize.assert_called_once_with(
        source_agent,
        ROWS[:2],
        base_url="http://127.0.0.1:9/apis/inference-gateway/v2/workspaces/staging/openai/-/v1",
        api_key="not-used",
        settings=StrandsHarnessConfig(dataset="dataset.jsonl", epochs=3, max_samples=2),
    )
    saved = tmp_path / "job-results" / RESULT_NAME
    assert yaml.safe_load((saved / OPTIMIZED_AGENT_FILENAME).read_text(encoding="utf-8")) == {
        **source_agent,
        "instructions": {"system": {"content": "new prompt"}},
    }
    assert json.loads((saved / SUMMARY_FILENAME).read_text(encoding="utf-8")) == {
        "strategy": "strands-harness-optimizer",
        "agent": "team/calculator-agent",
        "dataset_size": 2,
        "epochs": 3,
        "epoch_stats": [{"epoch": 1}],
        "original_prompt": "old prompt",
        "optimized_prompt": "new prompt",
        "optimized_agent_config": OPTIMIZED_AGENT_FILENAME,
    }


def test_run_rejects_dataset_outside_bundle(ctx: JobContext, bundle: Callable[..., None], agents: MagicMock) -> None:
    bundle(dataset="../elsewhere.jsonl")

    with pytest.raises(ValueError, match="outside"):
        run_job(ctx)


def test_run_missing_agent(ctx: JobContext, bundle: Callable[..., None], agents: MagicMock) -> None:
    bundle()
    agents.get_agent.side_effect = NotFoundError(httpx.Response(404, json={}, request=httpx.Request("GET", "http://x")))

    with pytest.raises(LocalRunError, match="does not exist"):
        run_job(ctx)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"config_format": "fabric.agent/v1alpha1"}, "nemo-agents-spec-v1"),
        ({"instructions": {}}, "no instructions"),
        ({"instructions": {"system": {"content": "  \n"}}}, "no instructions"),
        (
            {"models": {"default": {"provider": "openai", "model": "m", "base_url": "https://api.openai.com/v1"}}},
            "Gateway",
        ),
    ],
)
def test_run_refuses_agent(
    ctx: JobContext,
    bundle: Callable[..., None],
    agents: MagicMock,
    source_agent: dict[str, Any],
    override: dict[str, Any],
    message: str,
) -> None:
    bundle()
    source_agent.update(override)

    with pytest.raises(LocalRunError, match=message):
        run_job(ctx)

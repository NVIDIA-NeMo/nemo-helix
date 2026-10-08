# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``strands-harness-optimizer`` strategy job: its spec, its compiled step, and its run."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

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
from strands_harness_optimizer_plugin.jobs import optimize as optimize_module
from strands_harness_optimizer_plugin.jobs.optimize import (
    OPTIMIZED_AGENT_FILENAME,
    RESULT_NAME,
    SUMMARY_FILENAME,
    TASK_MODULE,
    StrandsHarnessOptimizeJob,
)
from strands_harness_optimizer_plugin.schemas.optimize import StrandsHarnessOptimizeSpec

pytest.importorskip("strands_harness_optimizer")

from strands_harness_optimizer_plugin import strands_bridge  # noqa: E402
from strands_harness_optimizer_plugin.strands_bridge import Outcome  # noqa: E402

SUBPROCESS_PROFILE = SubprocessJobExecutionProfile(profile="default")
CPU_PROFILE = DockerJobExecutionProfile(provider="cpu", profile="default", config=DockerJobExecutionProfileConfig())
STAGED: dict[str, Any] = {
    "agent": "calculator-agent",
    "optimize_config": "strands-harness-optimizer.yaml",
    "optimize_config_fileset": "default/sho-bundle",
}
ROWS = [{"input": f"What is {i} + {i}?", "expected_output": str(2 * i)} for i in range(4)]


def spec(**overrides: Any) -> StrandsHarnessOptimizeSpec:
    return StrandsHarnessOptimizeSpec.model_validate({**STAGED, **overrides})


@contextlib.contextmanager
def profiles(*execution_profiles: Any) -> Iterator[None]:
    async def _get_execution_profiles() -> Any:
        return SimpleNamespace(data=lambda: list(execution_profiles))

    client = MagicMock()
    client.get_execution_profiles = _get_execution_profiles
    with patch.object(optimize_module, "client_from_platform", return_value=client):
        yield


async def compile_spec(profile: str | None = None) -> Any:
    return await StrandsHarnessOptimizeJob.compile(
        workspace="staging",
        spec=spec(),
        entity_client=MagicMock(),
        job_name=None,
        async_sdk=MagicMock(),
        profile=profile,
    )


def run_job(ctx: JobContext, **overrides: Any) -> dict[str, Any]:
    return StrandsHarnessOptimizeJob().run({**STAGED, "workspace": "default", **overrides}, ctx=ctx, sdk=MagicMock())


@pytest.fixture
def bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Path]:
    """Stage a YAML + dataset in place of the fileset download; returns the bundle root."""

    def _stage(rows: list[dict[str, Any]] = ROWS, **settings: Any) -> Path:
        root = tmp_path / "bundle"
        root.mkdir()
        (root / "dataset.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        yaml_path = root / "strands-harness-optimizer.yaml"
        yaml_path.write_text(yaml.safe_dump({"dataset": "dataset.jsonl", **settings}), encoding="utf-8")

        @contextlib.contextmanager
        def fake_resolve(config_rel_path: str, fileset_ref: str | None, **kwargs: Any) -> Iterator[Path]:
            yield yaml_path

        monkeypatch.setattr(optimize_module, "resolve_staged_config", fake_resolve)
        return root

    return _stage


@pytest.fixture
def stored_agent(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, Any] | BaseException], MagicMock]:
    def _store(config: dict[str, Any] | BaseException) -> MagicMock:
        agents = MagicMock()
        if isinstance(config, BaseException):
            agents.get_agent.side_effect = config
        else:
            agents.get_agent.return_value.data.return_value = SimpleNamespace(config=config)
        monkeypatch.setattr(optimize_module, "client_from_platform", lambda sdk, cls: agents)
        return agents

    return _store


@pytest.fixture
def fake_optimize(monkeypatch: pytest.MonkeyPatch) -> list[SimpleNamespace]:
    calls: list[SimpleNamespace] = []

    def fake(config: dict[str, Any], rows: list[dict[str, Any]], **kwargs: Any) -> Outcome:
        calls.append(SimpleNamespace(config=config, rows=rows, **kwargs))
        return Outcome(
            original_prompt=config["instructions"]["system"]["content"],
            optimized_prompt="new prompt",
            epoch_stats=[{"epoch": 1, "avg_reward": 0.5}],
        )

    monkeypatch.setattr(strands_bridge, "optimize_system_prompt", fake)
    monkeypatch.setattr(optimize_module, "platform_auth_proxy", contextlib.nullcontext)
    return calls


@pytest.mark.parametrize("missing", ["agent", "optimize_config", "optimize_config_fileset"])
def test_spec_requires(missing: str) -> None:
    with pytest.raises(ValidationError, match=missing):
        StrandsHarnessOptimizeSpec.model_validate({k: v for k, v in STAGED.items() if k != missing})


@pytest.mark.parametrize(
    "override",
    [{"agent": "https://example.com/agent"}, {"optimize_config_fileset": "too/many/parts"}, {"output": "default/out"}],
)
def test_spec_rejects(override: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match=next(iter(override))):
        spec(**override)


async def test_compile_prefers_subprocess_and_stamps_the_spec() -> None:
    with profiles(SUBPROCESS_PROFILE, CPU_PROFILE):
        (step,) = (await compile_spec()).steps

    assert step.name == "strands-harness-optimizer"
    assert step.executor.provider == "subprocess"
    assert step.executor.command == ["python", "-m", TASK_MODULE]
    assert step.config == {**STAGED, "workspace": "staging"}


async def test_compile_falls_back_to_the_cpu_container() -> None:
    with profiles(CPU_PROFILE), patch.object(optimize_module, "get_qualified_image", return_value="reg/nhx-tasks:t"):
        (step,) = (await compile_spec(profile="default")).steps

    assert step.executor.provider == "cpu"
    assert step.executor.container.image == "reg/nhx-tasks:t"
    assert step.executor.container.command == [TASK_MODULE]


async def test_compile_reports_available_profiles_when_none_match() -> None:
    with profiles(DockerJobExecutionProfile(provider="gpu", profile="a100", config=DockerJobExecutionProfileConfig())):
        with pytest.raises(HelixJobCompilationError, match=r"Available profiles: \['gpu/a100'\]"):
            await compile_spec()


async def test_compile_is_retryable_when_jobs_is_unreachable() -> None:
    async def _boom() -> Any:
        raise NemoTransportError(httpx.ConnectError("refused", request=httpx.Request("GET", "http://x")))

    client = MagicMock()
    client.get_execution_profiles = _boom
    with patch.object(optimize_module, "client_from_platform", return_value=client):
        with pytest.raises(HelixJobDependencyUnavailableError, match="temporarily unavailable"):
            await compile_spec()


def test_run_writes_the_optimized_agent_and_registers_the_result(
    ctx: JobContext,
    tmp_path: Path,
    bundle: Callable[..., Path],
    stored_agent: Any,
    fake_optimize: list,
    source_agent: dict,
) -> None:
    bundle(epochs=3, max_samples=2)
    stored_agent(source_agent)

    result = run_job(ctx)

    assert result == {
        "status": "completed",
        "strategy": "strands-harness-optimizer",
        "agent": "default/calculator-agent",
        "optimized_prompt": "new prompt",
        "result": result["result"],
    }
    assert result["result"]["name"] == RESULT_NAME
    (call,) = fake_optimize
    assert call.config == source_agent
    assert call.rows == ROWS[:2]
    assert call.base_url.endswith("/workspaces/default/openai/-/v1")
    assert call.api_key == "not-used"
    assert call.settings.epochs == 3

    saved = tmp_path / "job-results" / RESULT_NAME
    optimized = yaml.safe_load((saved / OPTIMIZED_AGENT_FILENAME).read_text(encoding="utf-8"))
    assert optimized == {**source_agent, "instructions": {"system": {"content": "new prompt"}}}
    summary = json.loads((saved / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["dataset_size"] == 2
    assert summary["epochs"] == 3
    assert summary["epoch_stats"] == [{"epoch": 1, "avg_reward": 0.5}]
    assert (summary["original_prompt"], summary["optimized_prompt"]) == ("old prompt", "new prompt")


def test_run_routes_inference_through_the_auth_proxy(
    ctx: JobContext,
    bundle: Callable[..., Path],
    stored_agent: Any,
    fake_optimize: list,
    source_agent: dict,
    monkeypatch: Any,
) -> None:
    bundle()
    stored_agent(source_agent)
    monkeypatch.setattr(optimize_module, "platform_auth_proxy", lambda: contextlib.nullcontext("http://127.0.0.1:9"))

    run_job(ctx, agent="team/calculator-agent", workspace="staging")

    (call,) = fake_optimize
    assert call.base_url == "http://127.0.0.1:9/apis/inference-gateway/v2/workspaces/staging/openai/-/v1"


def test_run_rejects_a_dataset_outside_the_bundle(
    ctx: JobContext, bundle: Callable[..., Path], stored_agent: Any, fake_optimize: list, source_agent: dict
) -> None:
    bundle(dataset="../elsewhere.jsonl")
    stored_agent(source_agent)

    with pytest.raises(ValueError, match="outside"):
        run_job(ctx)


def test_run_reports_a_missing_agent_plainly(
    ctx: JobContext, bundle: Callable[..., Path], stored_agent: Any, fake_optimize: list
) -> None:
    bundle()
    stored_agent(NotFoundError(httpx.Response(404, json={}, request=httpx.Request("GET", "http://x"))))

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
def test_run_refuses_an_agent_it_cannot_optimize(
    ctx: JobContext,
    bundle: Callable[..., Path],
    stored_agent: Any,
    fake_optimize: list,
    source_agent: dict,
    override: dict,
    message: str,
) -> None:
    bundle()
    stored_agent({**source_agent, **override})

    with pytest.raises(LocalRunError, match=message):
        run_job(ctx)

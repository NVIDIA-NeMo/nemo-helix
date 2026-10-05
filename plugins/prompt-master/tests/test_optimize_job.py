# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``prompt-master`` strategy job: its spec, its compiled step, and its run."""

from __future__ import annotations

import contextlib
import copy
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
from prompt_master_plugin.jobs import optimize as optimize_module
from prompt_master_plugin.jobs.optimize import (
    OPTIMIZED_AGENT_FILENAME,
    RESULT_NAME,
    SUMMARY_FILENAME,
    TASK_MODULE,
    PromptMasterOptimizeJob,
)
from prompt_master_plugin.schemas.optimize import PromptMasterOptimizeSpec
from pydantic import ValidationError

SUBPROCESS_PROFILE = SubprocessJobExecutionProfile(profile="default")
CPU_PROFILE = DockerJobExecutionProfile(provider="cpu", profile="default", config=DockerJobExecutionProfileConfig())

STAGED: dict[str, Any] = {
    "optimize_config": "prompt-master.yaml",
    "optimize_config_fileset": "default/pm-bundle",
    "agent": "calculator-agent",
}


@contextlib.contextmanager
def profiles(*execution_profiles: Any) -> Iterator[None]:
    """Patch the Jobs client so ``compile`` sees exactly *execution_profiles*."""

    async def _get_execution_profiles() -> Any:
        return SimpleNamespace(data=lambda: list(execution_profiles))

    client = MagicMock()
    client.get_execution_profiles = _get_execution_profiles
    with patch.object(optimize_module, "client_from_platform", return_value=client):
        yield


def staged_spec(**overrides: Any) -> PromptMasterOptimizeSpec:
    return PromptMasterOptimizeSpec.model_validate({**STAGED, **overrides})


async def compile_spec(
    spec: PromptMasterOptimizeSpec, *, workspace: str = "default", profile: str | None = None
) -> Any:
    return await PromptMasterOptimizeJob.compile(
        workspace=workspace,
        spec=spec,
        entity_client=MagicMock(),
        job_name=None,
        async_sdk=MagicMock(),
        profile=profile,
    )


def run_job(ctx: JobContext, **spec: Any) -> dict[str, Any]:
    return PromptMasterOptimizeJob().run(
        {"agent": "calculator-agent", "workspace": "default", **spec}, ctx=ctx, sdk=MagicMock()
    )


@pytest.fixture
def stage_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Fake the fileset download so ``resolve_staged_config`` yields *overrides*; returns the arguments it saw."""

    def _stage(overrides: dict[str, Any]) -> dict[str, Any]:
        staged = tmp_path / "prompt-master.yaml"
        staged.write_text(yaml.safe_dump(overrides), encoding="utf-8")
        seen: dict[str, Any] = {}

        @contextlib.contextmanager
        def fake_resolve(config_rel_path: str, fileset_ref: str | None, **kwargs: Any) -> Iterator[Path]:
            seen.update(config_rel_path=config_rel_path, fileset_ref=fileset_ref, **kwargs)
            yield staged

        monkeypatch.setattr(optimize_module, "resolve_staged_config", fake_resolve)
        return seen

    return _stage


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------


def test_spec_requires_an_agent() -> None:
    with pytest.raises(ValidationError, match="agent"):
        PromptMasterOptimizeSpec.model_validate({k: v for k, v in STAGED.items() if k != "agent"})


def test_spec_rejects_a_url_as_the_agent() -> None:
    with pytest.raises(ValidationError, match="agent"):
        staged_spec(agent="https://example.com/agent")


def test_spec_config_is_optional() -> None:
    assert PromptMasterOptimizeSpec(agent="calculator-agent").optimize_config is None


@pytest.mark.parametrize("missing", ["optimize_config", "optimize_config_fileset"])
def test_spec_requires_config_and_fileset_together(missing: str) -> None:
    with pytest.raises(ValidationError, match="given together"):
        PromptMasterOptimizeSpec.model_validate({k: v for k, v in STAGED.items() if k != missing})


def test_spec_rejects_a_malformed_fileset_ref() -> None:
    with pytest.raises(ValidationError, match="optimize_config_fileset"):
        staged_spec(optimize_config_fileset="too/many/parts")


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------


async def test_compile_prefers_the_subprocess_executor_and_stamps_the_spec() -> None:
    with profiles(SUBPROCESS_PROFILE, CPU_PROFILE):
        job_spec = await compile_spec(staged_spec(output="default/pm-results"), workspace="staging")

    (step,) = job_spec.steps
    assert step.name == "prompt-master"
    assert step.executor.provider == "subprocess"
    assert step.executor.command == ["python", "-m", TASK_MODULE]
    assert step.config == {**STAGED, "output": "default/pm-results", "workspace": "staging"}


async def test_compile_falls_back_to_the_cpu_container() -> None:
    with (
        profiles(CPU_PROFILE),
        patch.object(optimize_module, "get_qualified_image", return_value="reg.example/nhx-tasks:test"),
    ):
        job_spec = await compile_spec(staged_spec(), profile="default")

    (step,) = job_spec.steps
    assert step.executor.provider == "cpu"
    assert step.executor.container.image == "reg.example/nhx-tasks:test"
    assert step.executor.container.entrypoint == ["python", "-m"]
    assert step.executor.container.command == [TASK_MODULE]


async def test_compile_reports_available_profiles_when_none_match() -> None:
    with profiles(DockerJobExecutionProfile(provider="gpu", profile="a100", config=DockerJobExecutionProfileConfig())):
        with pytest.raises(HelixJobCompilationError, match=r"Available profiles: \['gpu/a100'\]"):
            await compile_spec(staged_spec())


async def test_compile_is_retryable_when_jobs_is_unreachable() -> None:
    async def _boom() -> Any:
        raise NemoTransportError(httpx.ConnectError("connection refused", request=httpx.Request("GET", "http://x")))

    client = MagicMock()
    client.get_execution_profiles = _boom
    with (
        patch.object(optimize_module, "client_from_platform", return_value=client),
        pytest.raises(HelixJobDependencyUnavailableError, match="temporarily unavailable"),
    ):
        await compile_spec(staged_spec())


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_writes_the_optimized_agent_and_registers_the_result(
    ctx: JobContext,
    tmp_path: Path,
    fake_prompt_master: list[Any],
    stored_agent: Any,
    source_agent: dict[str, Any],
) -> None:
    with stored_agent(source_agent):
        result = run_job(ctx)

    assert result["status"] == "completed"
    assert result["strategy"] == "prompt-master"
    assert result["agent"] == "default/calculator-agent"
    assert result["optimized_prompt"] == "new prompt"
    assert result["result"]["name"] == RESULT_NAME
    assert "output" not in result

    optimizer, agent_config, base_dir = fake_prompt_master[0]
    assert optimizer.models["default"].base_url.endswith("/workspaces/default/openai/-/v1")
    assert agent_config["instructions"]["system"]["content"] == "old prompt"
    assert base_dir.is_relative_to(ctx.storage.ephemeral)

    saved = tmp_path / "job-results" / RESULT_NAME
    optimized = yaml.safe_load((saved / OPTIMIZED_AGENT_FILENAME).read_text(encoding="utf-8"))
    assert optimized == {**source_agent, "instructions": {"system": {"content": "new prompt"}}}
    summary = json.loads((saved / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["original_prompt"] == "old prompt"
    assert summary["optimized_prompt"] == "new prompt"
    assert summary["optimizer_model"] == optimizer.models["default"].model
    assert "Tightened scope" in summary["response"]


def test_run_leaves_the_stored_config_untouched(
    ctx: JobContext, fake_prompt_master: list[Any], stored_agent: Any, source_agent: dict[str, Any]
) -> None:
    before = copy.deepcopy(source_agent)

    with stored_agent(source_agent):
        run_job(ctx)

    assert source_agent == before


def test_run_resolves_the_agent_workspace_from_the_ref(
    ctx: JobContext, fake_prompt_master: list[Any], stored_agent: Any, source_agent: dict[str, Any]
) -> None:
    with stored_agent(source_agent) as agents:
        result = run_job(ctx, agent="team/calculator-agent")

    agents.get_agent.assert_called_once_with(name="calculator-agent", workspace="team")
    assert result["agent"] == "team/calculator-agent"


def test_run_publishes_to_a_local_output_dir(
    ctx: JobContext, tmp_path: Path, fake_prompt_master: list[Any], stored_agent: Any, source_agent: dict[str, Any]
) -> None:
    out = tmp_path / "out"

    with stored_agent(source_agent):
        result = run_job(ctx, output=str(out))

    assert result["output"] == {"type": "local_dir", "path": str(out.resolve())}
    assert (out / OPTIMIZED_AGENT_FILENAME).is_file()
    assert (out / SUMMARY_FILENAME).is_file()


def test_run_publishes_to_a_fileset(
    ctx: JobContext,
    fake_prompt_master: list[Any],
    stored_agent: Any,
    source_agent: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploaded: dict[str, Any] = {}

    def fake_upload(local_dir: Path, *, fileset: str, workspace: str, sdk: Any) -> None:
        uploaded.update(fileset=fileset, workspace=workspace, files=sorted(p.name for p in local_dir.iterdir()))

    monkeypatch.setattr(optimize_module, "upload_to_fileset", fake_upload)

    with stored_agent(source_agent):
        result = run_job(ctx, output="team/pm-results")

    assert result["output"] == {"type": "fileset", "fileset": "team/pm-results"}
    assert uploaded == {
        "fileset": "pm-results",
        "workspace": "team",
        "files": sorted([OPTIMIZED_AGENT_FILENAME, SUMMARY_FILENAME]),
    }


def test_run_merges_the_staged_config_into_the_optimizer(
    ctx: JobContext,
    stage_config: Callable[[dict[str, Any]], dict[str, Any]],
    fake_prompt_master: list[Any],
    stored_agent: Any,
    source_agent: dict[str, Any],
) -> None:
    seen = stage_config({"models": {"default": {"model": "gpt-5.6"}}, "runtime": {"timeout_seconds": 30}})

    with stored_agent(source_agent):
        result = run_job(ctx, **STAGED, workspace="staging")

    assert result["status"] == "completed"
    assert seen["config_rel_path"] == "prompt-master.yaml"
    assert seen["fileset_ref"] == "default/pm-bundle"
    assert seen["workspace"] == "staging"
    optimizer = fake_prompt_master[0][0]
    assert optimizer.models["default"].model == "gpt-5.6"
    assert optimizer.models["default"].provider == "nvidia"
    assert optimizer.runtime.timeout_seconds == 30
    assert optimizer.models["default"].base_url.endswith("/workspaces/staging/openai/-/v1")


def test_run_rejects_an_invalid_optimizer_config(
    ctx: JobContext, stage_config: Callable[[dict[str, Any]], dict[str, Any]]
) -> None:
    stage_config({"models": {"default": {"bogus": 1}}})

    with pytest.raises(ValidationError, match="bogus"):
        run_job(ctx, **STAGED)


def test_run_reports_a_missing_agent_plainly(ctx: JobContext, fake_prompt_master: list[Any], stored_agent: Any) -> None:
    missing = NotFoundError(httpx.Response(404, json={"detail": "not found"}, request=httpx.Request("GET", "http://x")))

    with stored_agent(missing), pytest.raises(LocalRunError, match="does not exist"):
        run_job(ctx)


@pytest.mark.parametrize(
    "config", [{}, {"schema_version": "fabric.agent/v1alpha1", "instructions": {"system": {"content": "p"}}}]
)
def test_run_refuses_an_agent_that_is_not_spec_v1(
    ctx: JobContext, fake_prompt_master: list[Any], stored_agent: Any, config: dict[str, Any]
) -> None:
    with stored_agent(config), pytest.raises(LocalRunError, match="nemo-agents-spec-v1"):
        run_job(ctx)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``prompt-master`` strategy job: its spec, its compiled step, and its run."""

from __future__ import annotations

import contextlib
import copy
import json
from collections.abc import Iterator
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
from prompt_master_plugin.config import PromptMasterConfigError
from prompt_master_plugin.jobs import optimize as optimize_module
from prompt_master_plugin.jobs.optimize import (
    OPTIMIZED_AGENT_FILENAME,
    RESULT_NAME,
    SUMMARY_FILENAME,
    TASK_MODULE,
    PromptMasterOptimizeJob,
)
from prompt_master_plugin.schemas.optimize import PromptMasterOptimizeSpec, PromptMasterOptimizeSubmitSpec
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


def run_job(ctx: JobContext, config_path: Path, **overrides: Any) -> dict[str, Any]:
    spec = {"optimize_config": str(config_path), "agent": "calculator-agent", "workspace": "default", **overrides}
    return PromptMasterOptimizeJob().run(spec, ctx=ctx, sdk=MagicMock())


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------


def test_submit_spec_requires_a_fileset_for_remote_submissions() -> None:
    with pytest.raises(ValidationError, match="optimize_config_fileset is required"):
        PromptMasterOptimizeSubmitSpec.model_validate({"optimize_config": "/abs/prompt-master.yaml", "agent": "a"})


def test_submit_spec_allows_an_absolute_host_path_for_local_runs() -> None:
    spec = PromptMasterOptimizeSubmitSpec.model_validate(
        {"optimize_config": "/abs/prompt-master.yaml", "agent": "a"}, context={"is_local": True}
    )

    assert spec.optimize_config_fileset is None


def test_spec_requires_an_agent() -> None:
    with pytest.raises(ValidationError, match="agent"):
        PromptMasterOptimizeSpec.model_validate({k: v for k, v in STAGED.items() if k != "agent"})


def test_spec_rejects_a_url_as_the_agent() -> None:
    with pytest.raises(ValidationError, match="agent"):
        staged_spec(agent="https://example.com/agent")


@pytest.mark.parametrize(
    "config_path", ["/abs/prompt-master.yaml", "../escape.yaml", "~/prompt-master.yaml", "C:\\bundle\\pm.yaml"]
)
def test_spec_rejects_config_paths_that_escape_the_fileset(config_path: str) -> None:
    with pytest.raises(ValidationError, match="relative to the fileset root"):
        staged_spec(optimize_config=config_path)


def test_spec_rejects_a_malformed_fileset_ref() -> None:
    with pytest.raises(ValidationError, match="'name' or 'workspace/name'"):
        staged_spec(optimize_config_fileset="too/many/parts")


async def test_to_spec_stamps_the_workspace() -> None:
    submitted = PromptMasterOptimizeSubmitSpec.model_validate(STAGED)

    spec = await PromptMasterOptimizeJob.to_spec(
        submitted, workspace="staging", entity_client=MagicMock(), async_sdk=MagicMock(), is_local=False
    )

    assert isinstance(spec, PromptMasterOptimizeSpec)
    assert spec.workspace == "staging"
    assert spec.agent == "calculator-agent"


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
    assert step.config == {
        "optimize_config": "prompt-master.yaml",
        "optimize_config_fileset": "default/pm-bundle",
        "agent": "calculator-agent",
        "output": "default/pm-results",
        "workspace": "staging",
    }


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


async def test_compile_requires_a_staged_fileset() -> None:
    spec = PromptMasterOptimizeSpec(optimize_config="/abs/prompt-master.yaml", agent="calculator-agent")

    with pytest.raises(HelixJobCompilationError, match="optimize_config_fileset is required"):
        await compile_spec(spec)


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
    config_path: Path,
    tmp_path: Path,
    fake_prompt_master: list[Any],
    stored_agent: Any,
    source_agent: dict[str, Any],
) -> None:
    with stored_agent(source_agent):
        result = run_job(ctx, config_path)

    assert result["status"] == "completed"
    assert result["strategy"] == "prompt-master"
    assert result["agent"] == "default/calculator-agent"
    assert result["optimized_prompt"] == "new prompt"
    assert result["result"]["name"] == RESULT_NAME
    assert "output" not in result

    (config, agent_config, base_dir, workspace) = fake_prompt_master[0]
    assert config.model.model == "gpt-5.6"
    assert agent_config["instructions"]["system"]["content"] == "old prompt"
    assert base_dir.is_relative_to(ctx.storage.ephemeral)
    # The optimizer model is bound to the Inference Gateway of the workspace the job runs in.
    assert workspace == "default"

    saved = tmp_path / "job-results" / RESULT_NAME
    optimized = yaml.safe_load((saved / OPTIMIZED_AGENT_FILENAME).read_text(encoding="utf-8"))
    assert optimized["instructions"]["system"]["content"] == "new prompt"
    # Everything but the prompt is carried over, so the file registers as a complete agent.
    assert {k: v for k, v in optimized.items() if k != "instructions"} == {
        k: v for k, v in source_agent.items() if k != "instructions"
    }
    summary = json.loads((saved / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["agent"] == "default/calculator-agent"
    assert summary["original_prompt"] == "old prompt"
    assert summary["optimized_prompt"] == "new prompt"
    assert summary["optimizer_model"]["model"] == "gpt-5.6"
    assert "Tightened scope" in summary["response"]


def test_run_leaves_the_stored_config_untouched(
    ctx: JobContext, config_path: Path, fake_prompt_master: list[Any], stored_agent: Any, source_agent: dict[str, Any]
) -> None:
    before = copy.deepcopy(source_agent)

    with stored_agent(source_agent):
        run_job(ctx, config_path)

    assert source_agent == before


def test_run_resolves_the_agent_workspace_from_the_ref(
    ctx: JobContext, config_path: Path, fake_prompt_master: list[Any], stored_agent: Any, source_agent: dict[str, Any]
) -> None:
    with stored_agent(source_agent) as agents:
        result = run_job(ctx, config_path, agent="team/calculator-agent")

    agents.get_agent.assert_called_once_with(name="calculator-agent", workspace="team")
    assert result["agent"] == "team/calculator-agent"


def test_run_publishes_to_a_local_output_dir(
    ctx: JobContext,
    config_path: Path,
    tmp_path: Path,
    fake_prompt_master: list[Any],
    stored_agent: Any,
    source_agent: dict[str, Any],
) -> None:
    out = tmp_path / "out"

    with stored_agent(source_agent):
        result = run_job(ctx, config_path, output=str(out))

    assert result["output"] == {"type": "local_dir", "path": str(out.resolve())}
    assert (out / OPTIMIZED_AGENT_FILENAME).is_file()
    assert (out / SUMMARY_FILENAME).is_file()


def test_run_publishes_to_a_fileset(
    ctx: JobContext,
    config_path: Path,
    fake_prompt_master: list[Any],
    stored_agent: Any,
    source_agent: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploaded: dict[str, Any] = {}

    def fake_upload(local_dir: Path, *, fileset: str, workspace: str, sdk: Any) -> None:
        uploaded.update(
            local_dir=local_dir, fileset=fileset, workspace=workspace, files=sorted(p.name for p in local_dir.iterdir())
        )

    monkeypatch.setattr(optimize_module, "upload_to_fileset", fake_upload)

    with stored_agent(source_agent):
        result = run_job(ctx, config_path, output="team/pm-results")

    assert result["output"] == {"type": "fileset", "fileset": "team/pm-results"}
    assert uploaded["fileset"] == "pm-results"
    assert uploaded["workspace"] == "team"
    assert uploaded["files"] == sorted([OPTIMIZED_AGENT_FILENAME, SUMMARY_FILENAME])


def test_run_stages_the_config_from_the_fileset(
    ctx: JobContext,
    tmp_path: Path,
    fake_prompt_master: list[Any],
    stored_agent: Any,
    source_agent: dict[str, Any],
    optimizer_config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged" / "prompt-master.yaml"
    staged.parent.mkdir()
    staged.write_text(yaml.safe_dump(optimizer_config), encoding="utf-8")
    seen: dict[str, Any] = {}

    @contextlib.contextmanager
    def fake_resolve(config_rel_path: str, fileset_ref: str | None, **kwargs: Any) -> Iterator[Path]:
        seen.update(config_rel_path=config_rel_path, fileset_ref=fileset_ref, **kwargs)
        yield staged

    monkeypatch.setattr(optimize_module, "resolve_staged_config", fake_resolve)

    with stored_agent(source_agent):
        result = PromptMasterOptimizeJob().run({**STAGED, "workspace": "staging"}, ctx=ctx, sdk=MagicMock())

    assert result["status"] == "completed"
    assert seen["config_rel_path"] == "prompt-master.yaml"
    assert seen["fileset_ref"] == "default/pm-bundle"
    assert seen["workspace"] == "staging"
    assert seen["kind"] == "prompt-master-config"
    assert fake_prompt_master[0][3] == "staging"


def test_run_requires_a_platform_client(ctx: JobContext, config_path: Path) -> None:
    with pytest.raises(LocalRunError, match="platform client"):
        PromptMasterOptimizeJob().run(
            {"optimize_config": str(config_path), "agent": "calculator-agent", "workspace": "default"}, ctx=ctx
        )


def test_run_rejects_an_invalid_optimizer_config(
    ctx: JobContext, tmp_path: Path, stored_agent: Any, source_agent: dict[str, Any]
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: {}\n", encoding="utf-8")

    with stored_agent(source_agent), pytest.raises(PromptMasterConfigError, match="Invalid Prompt Master config"):
        run_job(ctx, bad)


def test_run_reports_a_missing_agent_plainly(
    ctx: JobContext, config_path: Path, fake_prompt_master: list[Any], stored_agent: Any
) -> None:
    missing = NotFoundError(httpx.Response(404, json={"detail": "not found"}, request=httpx.Request("GET", "http://x")))

    with stored_agent(missing), pytest.raises(LocalRunError, match="does not exist"):
        run_job(ctx, config_path)


def test_run_refuses_an_agent_in_another_config_format(
    ctx: JobContext, config_path: Path, fake_prompt_master: list[Any], stored_agent: Any
) -> None:
    fabric_package = {"schema_version": "fabric.agent/v1alpha1", "instructions": {"system": {"content": "p"}}}

    with stored_agent(fabric_package), pytest.raises(LocalRunError, match="config_format"):
        run_job(ctx, config_path)


def test_run_refuses_an_agent_with_an_empty_config(
    ctx: JobContext, config_path: Path, fake_prompt_master: list[Any], stored_agent: Any
) -> None:
    with stored_agent({}), pytest.raises(LocalRunError, match="empty or invalid"):
        run_job(ctx, config_path)

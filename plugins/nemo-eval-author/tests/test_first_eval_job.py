# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from nemo_eval_author_plugin.jobs import first_eval as job_module
from nemo_eval_author_plugin.jobs.first_eval import RESULT_NAME, SUMMARY_FILENAME, TASK_MODULE, FirstEvalJob
from nemo_eval_author_plugin.schemas.first_eval import FirstEvalSpec
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.errors import LocalRunError
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nemo_helix_plugin.jobs.execution_profiles import (
    DockerJobExecutionProfile,
    DockerJobExecutionProfileConfig,
    SubprocessJobExecutionProfile,
)
from pydantic import ValidationError

SUBPROCESS_PROFILE = SubprocessJobExecutionProfile(profile="default")
CPU_PROFILE = DockerJobExecutionProfile(provider="cpu", profile="default", config=DockerJobExecutionProfileConfig())
STAGED = {"author_config": "author.yaml", "author_config_fileset": "default/bundle"}


@contextlib.contextmanager
def profiles(*execution_profiles: Any) -> Iterator[None]:
    async def _get_execution_profiles() -> Any:
        return SimpleNamespace(data=lambda: list(execution_profiles))

    client = MagicMock()
    client.get_execution_profiles = _get_execution_profiles
    with patch.object(job_module, "client_from_platform", return_value=client):
        yield


async def compile_spec(profile: str | None = None) -> Any:
    return await FirstEvalJob.compile(
        workspace="staging",
        spec=FirstEvalSpec(agent="calculator-agent"),
        entity_client=MagicMock(),
        job_name=None,
        async_sdk=MagicMock(),
        profile=profile,
    )


def run_job(ctx: JobContext, **spec: Any) -> dict[str, Any]:
    return FirstEvalJob().run({"agent": "calculator-agent", "workspace": "default", **spec}, ctx=ctx, sdk=MagicMock())


@pytest.mark.parametrize(("payload", "match"), [({}, "agent"), ({"agent": "a", "strategy": "x"}, "strategy")])
def test_spec_rejects_invalid_payloads(payload: dict[str, str], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        FirstEvalSpec.model_validate(payload)


@pytest.mark.parametrize("missing", ["author_config", "author_config_fileset"])
def test_spec_requires_config_and_fileset_together(missing: str) -> None:
    with pytest.raises(ValidationError, match="given together"):
        FirstEvalSpec.model_validate({"agent": "a", **{k: v for k, v in STAGED.items() if k != missing}})


async def test_compile_prefers_the_subprocess_executor_and_stamps_the_workspace() -> None:
    with profiles(SUBPROCESS_PROFILE, CPU_PROFILE):
        (step,) = (await compile_spec()).steps

    assert step.name == "first-eval"
    assert step.executor.provider == "subprocess"
    assert step.executor.command == ["python", "-m", TASK_MODULE]
    assert step.config["agent"] == "calculator-agent"
    assert step.config["workspace"] == "staging"


async def test_compile_falls_back_to_the_cpu_container() -> None:
    with profiles(CPU_PROFILE), patch.object(job_module, "get_qualified_image", return_value="reg/nhx-tasks:t"):
        (step,) = (await compile_spec(profile="default")).steps

    assert step.executor.provider == "cpu"
    assert step.executor.container.image == "reg/nhx-tasks:t"
    assert step.executor.container.command == [TASK_MODULE]


async def test_compile_reports_available_profiles_when_none_match() -> None:
    with profiles(DockerJobExecutionProfile(provider="gpu", profile="a100", config=DockerJobExecutionProfileConfig())):
        with pytest.raises(HelixJobCompilationError, match=r"Available profiles: \['gpu/a100'\]"):
            await compile_spec()


def test_run_stages_inputs_and_saves_results(
    ctx: JobContext, tmp_path: Path, platform: Callable[..., Any], fake_author: list[dict[str, Any]]
) -> None:
    seen = platform()

    result = run_job(ctx)

    workspace = fake_author[0]["base_dir"] / "workspace"
    assert fake_author[0]["base_dir"] == ctx.storage.ephemeral / "eval-author" / "fabric"
    assert "default/calculator-agent" in fake_author[0]["input"]
    assert fake_author[0]["author"].models["default"].base_url.endswith("/workspaces/default/openai/-/v1")
    assert yaml.safe_load((workspace / "agent.yaml").read_text(encoding="utf-8")) == seen.config
    assert (workspace / ".agents" / "skills" / "eval-author-first-eval" / "SKILL.md").is_file()
    assert seen.downloads == []

    saved = tmp_path / "job-results" / RESULT_NAME
    files = [".eval-author/first-eval.md", ".eval-author/task-drafts/add.md", "ETHOS.md"]
    assert all((saved / name).is_file() for name in files)
    summary = json.loads((saved / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary == {
        "agent": "default/calculator-agent",
        "author_model": "nvidia-nemotron-3-super-120b-a12b",
        "output_fileset": "default/calculator-agent-evals",
        "files": files,
        "response": "Authored one case.",
    }
    ((local_dir, upload),) = seen.uploads
    assert local_dir == ctx.storage.ephemeral / "eval-author" / "results"
    assert (upload["workspace"], upload["fileset"]) == ("default", "calculator-agent-evals")
    assert result["status"] == "completed"
    assert result["output_fileset"] == "default/calculator-agent-evals"
    assert result["files"] == files
    assert result["result"]["name"] == RESULT_NAME


def test_run_stages_the_ethos_fileset_when_it_exists(
    ctx: JobContext, platform: Callable[..., Any], fake_author: list[dict[str, Any]]
) -> None:
    seen = platform(ethos=SimpleNamespace(name="calculator-agent-ethos"))

    run_job(ctx, agent="team/calculator-agent")

    seen.agents.get_agent.assert_called_once_with(name="calculator-agent", workspace="team")
    assert seen.downloads == [("team", "calculator-agent-ethos", fake_author[0]["base_dir"] / "workspace")]
    assert seen.uploads[0][1]["workspace"] == "team"


def test_run_honours_explicit_filesets(
    ctx: JobContext, platform: Callable[..., Any], fake_author: list[dict[str, Any]]
) -> None:
    seen = platform()

    result = run_job(ctx, agent_fileset="docs/calculator-repo", output="evals/suite")

    assert seen.downloads[0][:2] == ("docs", "calculator-repo")
    assert (seen.uploads[0][1]["workspace"], seen.uploads[0][1]["fileset"]) == ("evals", "suite")
    assert result["output_fileset"] == "evals/suite"


def test_run_merges_the_staged_author_config(
    ctx: JobContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: Callable[..., Any], fake_author: list
) -> None:
    staged = tmp_path / "author.yaml"
    staged.write_text(yaml.safe_dump({"models": {"default": {"model": "gpt-5.6"}}, "runtime": {"timeout_seconds": 30}}))

    @contextlib.contextmanager
    def fake_resolve(config_rel_path: str, fileset_ref: str | None, **kwargs: Any) -> Iterator[Path]:
        assert (config_rel_path, fileset_ref, kwargs["workspace"]) == ("author.yaml", "default/bundle", "default")
        yield staged

    monkeypatch.setattr(job_module, "resolve_staged_config", fake_resolve)
    platform()

    run_job(ctx, **STAGED)

    author = fake_author[0]["author"]
    assert (author.models["default"].model, author.models["default"].provider) == ("gpt-5.6", "nvidia")
    assert author.runtime.timeout_seconds == 30


def test_run_fails_when_the_author_produced_nothing(
    ctx: JobContext, monkeypatch: pytest.MonkeyPatch, platform: Callable[..., Any]
) -> None:
    monkeypatch.setattr(job_module, "run_author", lambda author, *, input, base_dir: "I asked a question instead.")
    platform()

    with pytest.raises(LocalRunError, match=r"no \.eval-author/ directory"):
        run_job(ctx)


def test_run_reports_a_missing_agent_plainly(ctx: JobContext, platform: Callable[..., Any]) -> None:
    platform(agent=NotFoundError(httpx.Response(404, json={}, request=httpx.Request("GET", "http://x"))))

    with pytest.raises(LocalRunError, match="does not exist"):
        run_job(ctx)


def test_job_is_not_an_optimization_strategy() -> None:
    assert not hasattr(FirstEvalJob, "nemo_agent_optimization_strategy")

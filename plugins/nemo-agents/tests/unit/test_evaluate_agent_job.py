# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``EvaluateAgentJob`` platform-managed-job paths."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nemo_agents_plugin.jobs.evaluate_agent import EvaluateAgentJob, EvaluateAgentSpec
from nemo_agents_plugin.refs import AgentRef
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.refs import FilesetRef


@pytest.mark.asyncio
async def test_compile_produces_single_cpu_step() -> None:
    spec = EvaluateAgentSpec(
        agent=AgentRef("calc"),
        eval_config="config.yml",
        eval_config_fileset=FilesetRef("nemo-agent-eval-calc"),
        output=FilesetRef("nemo-agent-eval-calc"),
        workspace="default",
    )
    platform_spec = await EvaluateAgentJob.compile(
        workspace="default",
        spec=spec,
        entity_client=MagicMock(),
        job_name=None,
        async_sdk=MagicMock(),
    )
    steps = list(platform_spec["steps"])
    assert len(steps) == 1
    step = steps[0]
    assert step["name"] == "evaluate-agent"
    assert step["executor"]["provider"] == "subprocess"
    assert step["executor"]["command"] == ["python", "-m", "nemo_agents_plugin.tasks.evaluate"]
    assert step["config"]["agent"] == "calc"
    assert step["config"]["eval_config"] == "config.yml"
    assert step["config"]["eval_config_fileset"] == "nemo-agent-eval-calc"

    from nemo_platform_plugin.jobs.constants import (
        DEFAULT_JOB_STORAGE_PATH,
        EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
        PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
    )

    env = {e["name"]: e["value"] for e in step["environment"]}
    # Persistent path is declared in the compiled step so the K8s backend
    # can wire up the volume mount; ephemeral path comes from the K8s
    # backend's base env list (no need to duplicate it here).
    assert env[PERSISTENT_JOB_STORAGE_PATH_ENVVAR] == DEFAULT_JOB_STORAGE_PATH
    assert EPHEMERAL_TASK_STORAGE_PATH_ENVVAR not in env


@pytest.mark.asyncio
async def test_compile_url_workspace_overrides_spec_workspace() -> None:
    spec = EvaluateAgentSpec(
        eval_config="config.yml",
        agent=AgentRef("calc"),
        workspace="research",
    )
    platform_spec = await EvaluateAgentJob.compile(
        workspace="staging",
        spec=spec,
        entity_client=MagicMock(),
        job_name=None,
        async_sdk=MagicMock(),
    )
    config = next(iter(platform_spec["steps"]))["config"]
    assert config["workspace"] == "staging"


def test_resolve_eval_config_local_path_pass_through(tmp_path: Path, ctx: JobContext) -> None:
    job = EvaluateAgentJob()
    spec = EvaluateAgentSpec(eval_config=str(tmp_path / "config.yml"), eval_config_fileset=None)
    with job._resolve_eval_config(spec, ctx=ctx, sdk=None) as resolved:
        assert resolved == Path(str(tmp_path / "config.yml"))


def test_resolve_eval_config_fileset_downloads_via_sdk(tmp_path: Path, ctx: JobContext) -> None:
    job = EvaluateAgentJob()
    spec = EvaluateAgentSpec(
        eval_config="config.yml",
        eval_config_fileset=FilesetRef("nemo-agent-eval-calc"),
        workspace="default",
    )

    sdk = MagicMock()
    manager = MagicMock()

    def _fake_download(_ref: str, *, local_dir: Path) -> None:
        Path(local_dir, "config.yml").write_text("eval: {}")

    manager.download_from_url.side_effect = _fake_download

    with (
        patch("nemo_agents_plugin.jobs.evaluate_agent.client_from_platform", return_value=MagicMock()),
        patch("nemo_agents_plugin.jobs.evaluate_agent.FilesetFileSystem", return_value=MagicMock()),
        patch("nemo_agents_plugin.jobs.evaluate_agent.FilesetFileManager", return_value=manager) as manager_cls,
    ):
        with job._resolve_eval_config(spec, ctx=ctx, sdk=sdk) as resolved:
            assert resolved.exists()
            assert resolved.name == "config.yml"
            assert resolved.read_text() == "eval: {}"

    manager_cls.assert_called_once()
    assert manager_cls.call_args.kwargs["fileset_name"] == "nemo-agent-eval-calc"
    assert manager_cls.call_args.kwargs["workspace"] == "default"
    manager.download_from_url.assert_called_once()
    assert manager.download_from_url.call_args.args == ("default/nemo-agent-eval-calc",)


def test_resolve_eval_config_fileset_without_sdk_raises(tmp_path: Path, ctx: JobContext) -> None:
    job = EvaluateAgentJob()
    spec = EvaluateAgentSpec(eval_config="config.yml", eval_config_fileset=FilesetRef("nemo-agent-eval-calc"))
    with pytest.raises(Exception) as exc:
        with job._resolve_eval_config(spec, ctx=ctx, sdk=None):
            pass
    assert "sdk" in str(exc.value).lower()


def test_resolve_output_fileset_uploads_on_clean_exit(tmp_path: Path, ctx: JobContext) -> None:
    job = EvaluateAgentJob()
    sdk = MagicMock()
    manager = MagicMock()

    with (
        patch("nemo_agents_plugin.jobs.evaluate_agent.client_from_platform", return_value=MagicMock()),
        patch("nemo_agents_plugin.jobs.evaluate_agent.FilesetFileSystem", return_value=MagicMock()),
        patch("nemo_agents_plugin.jobs.evaluate_agent.FilesetFileManager", return_value=manager) as manager_cls,
    ):
        with job._resolve_output(FilesetRef("eval-out"), workspace="default", ctx=ctx, sdk=sdk):
            pass

    manager_cls.assert_called_once()
    assert manager_cls.call_args.kwargs["fileset_name"] == "eval-out"
    assert manager_cls.call_args.kwargs["workspace"] == "default"
    assert manager_cls.call_args.kwargs["ensure_fileset_exists"] is True
    manager.validate_storage.assert_called_once_with()
    manager.upload.assert_called_once()
    assert manager.upload.call_args.kwargs["remote_path"] == ""


def test_resolve_output_fileset_skips_upload_when_body_raises(tmp_path: Path, ctx: JobContext) -> None:
    job = EvaluateAgentJob()
    sdk = MagicMock()

    with (
        patch.object(EvaluateAgentJob, "_upload_to_fileset") as upload,
        pytest.raises(RuntimeError, match="nat eval blew up"),
    ):
        with job._resolve_output(FilesetRef("eval-out"), workspace="default", ctx=ctx, sdk=sdk):
            raise RuntimeError("nat eval blew up")

    upload.assert_not_called()


def test_run_failed_subprocess_skips_fileset_upload(tmp_path: Path, ctx: JobContext) -> None:
    eval_yaml = tmp_path / "config.yml"
    eval_yaml.write_text("eval:\n  general:\n    output_dir: ./.tmp\n")

    sdk = MagicMock()
    spec = {
        "eval_config": str(eval_yaml),
        "agent": "calc",
        "output": "eval-out",
        "workspace": "default",
    }

    cpe = subprocess.CalledProcessError(returncode=2, cmd=["nat", "eval"])
    with (
        patch.object(EvaluateAgentJob, "_upload_to_fileset") as upload,
        patch("nemo_agents_plugin.jobs.evaluate_agent.subprocess.run", side_effect=cpe),
    ):
        result = EvaluateAgentJob().run(spec, ctx=ctx, sdk=sdk)

    assert result == {"status": "failed", "returncode": 2}
    upload.assert_not_called()


def test_run_subprocess_timeout_skips_fileset_upload(tmp_path: Path, ctx: JobContext) -> None:
    """A hung `nat eval` must surface as a timeout failure (returncode 124) and
    skip the fileset upload — same protection as a non-zero exit."""
    eval_yaml = tmp_path / "config.yml"
    eval_yaml.write_text("eval:\n  general:\n    output_dir: ./.tmp\n")

    sdk = MagicMock()
    spec = {
        "eval_config": str(eval_yaml),
        "agent": "calc",
        "output": "eval-out",
        "workspace": "default",
    }

    timeout = subprocess.TimeoutExpired(cmd=["nat", "eval"], timeout=3600)
    with (
        patch.object(EvaluateAgentJob, "_upload_to_fileset") as upload,
        patch("nemo_agents_plugin.jobs.evaluate_agent.subprocess.run", side_effect=timeout),
    ):
        result = EvaluateAgentJob().run(spec, ctx=ctx, sdk=sdk)

    assert result == {"status": "failed", "returncode": 124}
    upload.assert_not_called()


def test_resolve_output_no_output_uses_persistent_results(tmp_path: Path, ctx: JobContext) -> None:
    # No-output fallback writes under ``ctx.storage.persistent / "results"`` so
    # the platform-provisioned volume captures eval artifacts.
    job = EvaluateAgentJob()
    with job._resolve_output(None, workspace="default", ctx=ctx, sdk=None) as base:
        assert base == ctx.storage.persistent / "results"
        assert base.is_dir()


def test_resolve_eval_config_fileset_tempdir_lands_under_ctx_ephemeral(tmp_path: Path, ctx: JobContext) -> None:
    # Fileset download tempdir is created under ``ctx.storage.ephemeral`` so the
    # platform-injected scratch volume is used instead of $TMPDIR.
    job = EvaluateAgentJob()
    spec = EvaluateAgentSpec(
        eval_config="config.yml",
        eval_config_fileset=FilesetRef("nemo-agent-eval-calc"),
        workspace="default",
    )
    sdk = MagicMock()
    manager = MagicMock()

    seen: dict[str, Path] = {}

    def _fake_download(_ref: str, *, local_dir: Path) -> None:
        seen["local_path"] = local_dir
        Path(local_dir, "config.yml").write_text("eval: {}")

    manager.download_from_url.side_effect = _fake_download

    with (
        patch("nemo_agents_plugin.jobs.evaluate_agent.client_from_platform", return_value=MagicMock()),
        patch("nemo_agents_plugin.jobs.evaluate_agent.FilesetFileSystem", return_value=MagicMock()),
        patch("nemo_agents_plugin.jobs.evaluate_agent.FilesetFileManager", return_value=manager),
    ):
        with job._resolve_eval_config(spec, ctx=ctx, sdk=sdk):
            pass

    assert seen["local_path"].parent == ctx.storage.ephemeral


def test_resolve_output_fileset_tempdir_lands_under_ctx_ephemeral(tmp_path: Path, ctx: JobContext) -> None:
    # Same invariant for the fileset-output staging tempdir.
    job = EvaluateAgentJob()
    sdk = MagicMock()

    captured: dict[str, Path] = {}
    with patch.object(EvaluateAgentJob, "_upload_to_fileset"):
        with job._resolve_output(FilesetRef("eval-out"), workspace="default", ctx=ctx, sdk=sdk) as base:
            captured["base"] = base

    assert captured["base"].parent == ctx.storage.ephemeral

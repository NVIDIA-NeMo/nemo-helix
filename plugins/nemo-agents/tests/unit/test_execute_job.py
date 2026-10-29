# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable
from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast
from unittest.mock import AsyncMock, MagicMock, call, patch
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.config import AgentJobsConfig, AgentsConfig, DeploymentsRunnerConfig
from nemo_agents_plugin.entities import (
    Agent,
    AgentComputeSpec,
    AgentEnvironment,
    AgentEnvironmentInline,
    AgentEnvironmentSpec,
    AgentInline,
    ComputeResources,
    ComputeSpecInline,
    EnvironmentSpecInline,
    McpFulfillment,
)
from nemo_agents_plugin.fabric.runtime import FabricRuntimeResult
from nemo_agents_plugin.jobs import gateway_proxy
from nemo_agents_plugin.jobs.execute import (
    DEFAULT_AGENT_EXECUTION_TIMEOUT_SECONDS,
    FABRIC_ERROR_RESULT_NAME,
    FABRIC_RUN_RESULT_NAME,
    INPUT_WORKDIR_RESULT_NAME,
    OUTPUT_ARTIFACTS_RESULT_NAME,
    OUTPUT_WORKDIR_RESULT_NAME,
    ExecuteAgentJob,
    ExecuteAgentJobConfig,
    ExecuteAgentStepConfig,
    PostRunActionConfig,
    ResolvedAgentConfig,
    _configure_intake_telemetry,
    _log_agent_stderr,
    _stage_stored_agent_ethos,
)
from nemo_agents_plugin.runner.fabric_artifact_staging import (
    FabricArtifactStagingError,
    FabricEthosFilesetNotFound,
    stage_fabric_ethos_dir,
)
from nemo_agents_plugin.tasks.execute.workdir import (
    AgentWorkdir,
    AgentWorkdirArtifactMount,
    _canonical_files_ref,
    materialize_agent_workdir,
    validate_agent_workdir,
)
from nemo_agents_plugin.telemetry import intake_export
from nemo_agents_plugin.telemetry.intake_export import supports_intake_atif_export, wants_intake_atif_export
from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_helix_plugin.dependencies import get_entity_client, get_nemo_client
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.job_usage import LocalJobUsageReporter
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nemo_helix_plugin.jobs.routes import add_job_routes
from pydantic import BaseModel, ConfigDict, ValidationError

# The hermes adapter is not installed on Python 3.14 (see this plugin's pyproject.toml).
requires_hermes_adapter = pytest.mark.skipif(
    find_spec("nemo_fabric_adapters") is None or find_spec("nemo_fabric_adapters.hermes") is None,
    reason="needs the hermes harness adapter, which is not installed on Python 3.14",
)


class _OpenedAsyncClient:
    async def __aenter__(self) -> "_OpenedAsyncClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


def _sdk_opening_async_client() -> MagicMock:
    sdk = MagicMock()
    sdk.to_async.return_value = _OpenedAsyncClient()
    return sdk


@pytest.fixture(autouse=True)
def _ethos_fileset_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing ``run()`` tests pass a ``MagicMock`` SDK and declare no skills.

    They should keep seeing a missing ethos fileset: the real stager runs, the
    download is not found, and validation passes. Tests that need a fileset
    restore ``_stage_stored_agent_ethos`` and replace this download.
    """

    async def _missing(*_args: object, **_kwargs: object) -> None:
        raise FabricEthosFilesetNotFound("missing fileset")

    monkeypatch.setattr(
        "nemo_agents_plugin.runner.fabric_artifact_staging._download_fileset",
        _missing,
    )

    def _stage(
        sdk: object,
        *,
        workspace: str,
        agent_name: str,
        agent_config: AgentConfig,
        base_dir: Path,
    ) -> None:
        del sdk

        async def _run() -> None:
            await stage_fabric_ethos_dir(
                workspace=workspace,
                agent_name=agent_name,
                agent_config=agent_config.model_dump(mode="json"),
                base_dir=base_dir,
                files_client=AsyncMock(),
            )

        asyncio.run(_run())

    monkeypatch.setattr("nemo_agents_plugin.jobs.execute._stage_stored_agent_ethos", _stage)


def _use_real_ethos_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nemo_agents_plugin.jobs.execute._stage_stored_agent_ethos",
        _stage_stored_agent_ethos,
    )
    monkeypatch.setattr(
        "nemo_agents_plugin.jobs.execute.AsyncFilesClient.from_client",
        lambda _client: AsyncMock(),
    )


class _TypedFilesResponse:
    def __init__(self, data: list[object]) -> None:
        self._body = SimpleNamespace(data=data)

    def data(self) -> SimpleNamespace:
        return self._body


def _agent_config(**environment: str) -> dict[str, Any]:
    config = {
        "config_format": "nemo-agents-spec-v1",
        "name": "calc",
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes"}},
        "models": {"default": {"provider": "openai", "model": "openai/gpt-5.4"}},
    }
    if environment:
        config["environment"] = environment
    return config


def _agent(name: str = "calc", workspace: str = "default", config_format: str = "nemo-agents-spec-v1") -> Agent:
    config = _agent_config() if config_format == "nemo-agents-spec-v1" else {"workflow": {}}
    return Agent(name=name, workspace=workspace, config=config, config_format=config_format)


def test_run_passes_caller_context_with_platform_fields_under_reserved_key(ctx: JobContext) -> None:
    caller_context = {"run_scope": {"since": "2026-10-01"}, "evaluation_id": "eval-1"}
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello", context=caller_context),
        agent=_resolved_agent(workspace="research"),
    )
    seen: dict[str, Any] = {}

    async def _invoke(request: Any) -> FabricRuntimeResult:
        seen["context"] = request.caller_context
        return FabricRuntimeResult(status="succeeded")

    with patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert seen["context"] == {
        **caller_context,
        "nemo_helix": {"job_id": ctx.job_id, "job_workspace": ctx.workspace, "agent": "research/calc"},
    }


def test_run_without_caller_context_sends_only_platform_fields(ctx: JobContext) -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
    )
    seen: dict[str, Any] = {}

    async def _invoke(request: Any) -> FabricRuntimeResult:
        seen["context"] = request.caller_context
        return FabricRuntimeResult(status="succeeded")

    with patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert seen["context"] == {
        "nemo_helix": {"job_id": ctx.job_id, "job_workspace": ctx.workspace, "agent": "default/calc"},
    }


def test_job_config_rejects_reserved_context_key() -> None:
    with pytest.raises(ValidationError, match="'nemo_helix' is reserved"):
        ExecuteAgentJobConfig(agent="calc", input="hello", context={"nemo_helix": {"job_id": "spoofed"}})


@pytest.mark.asyncio
async def test_to_spec_keeps_caller_context_on_the_step_config() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _agent()

    spec = await ExecuteAgentJob.to_spec(
        ExecuteAgentJobConfig(agent="calc", input="hello", context={"evaluation_id": "eval-1"}),
        workspace="default",
        entity_client=entity_client,
        async_sdk=_sdk_with_files(),
        is_local=False,
    )

    step_config = ExecuteAgentStepConfig.model_validate(spec)
    assert step_config.request.context == {"evaluation_id": "eval-1"}


def _sdk_with_files(data: list[object] | None = None) -> MagicMock:
    sdk = MagicMock()
    sdk.files.list_files = AsyncMock(return_value=_TypedFilesResponse(data if data is not None else [object()]))
    return sdk


def _resolved_agent(name: str = "calc", workspace: str = "default") -> ResolvedAgentConfig:
    return ResolvedAgentConfig(
        name=name,
        workspace=workspace,
        config=_agent_config(),
        config_format="nemo-agents-spec-v1",
    )


def test_job_config_defaults_to_bounded_fabric_timeout() -> None:
    config = ExecuteAgentJobConfig(agent="calc", input="hello")

    assert config.timeout_seconds == DEFAULT_AGENT_EXECUTION_TIMEOUT_SECONDS


def test_job_config_rejects_non_positive_fabric_timeout() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        ExecuteAgentJobConfig(agent="calc", input="hello", timeout_seconds=0)


@pytest.mark.asyncio
async def test_to_spec_resolves_bare_agent_against_route_workspace() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _agent()

    spec = await ExecuteAgentJob.to_spec(
        ExecuteAgentJobConfig(agent="calc", input="hello"),
        workspace="default",
        entity_client=entity_client,
        async_sdk=_sdk_with_files(),
        is_local=False,
    )

    step_config = ExecuteAgentStepConfig.model_validate(spec)
    assert step_config.agent.name == "calc"
    assert step_config.agent.workspace == "default"
    assert step_config.agent_ref == "default/calc"
    assert step_config.request.input == "hello"
    assert step_config.request.timeout_seconds == DEFAULT_AGENT_EXECUTION_TIMEOUT_SECONDS
    assert step_config.workdir is None
    assert (
        step_config.agent.config["models"]["default"]["base_url"]
        == "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    )
    entity_client.get.assert_awaited_once_with(Agent, name="calc", workspace="default")


@pytest.mark.asyncio
async def test_to_spec_preserves_explicit_agent_workspace() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _agent(workspace="research")

    spec = await ExecuteAgentJob.to_spec(
        ExecuteAgentJobConfig(agent="research/calc", input="hello"),
        workspace="default",
        entity_client=entity_client,
        async_sdk=_sdk_with_files(),
        is_local=False,
    )

    entity_client.get.assert_awaited_once_with(Agent, name="calc", workspace="research")
    # A job in one workspace that runs another workspace's agent is listed under that agent's ref.
    assert ExecuteAgentStepConfig.model_validate(spec).agent_ref == "research/calc"


@pytest.mark.asyncio
async def test_to_spec_rejects_missing_agent() -> None:
    entity_client = AsyncMock()
    entity_client.get.side_effect = NemoEntityNotFoundError("missing")

    with pytest.raises(ValueError, match="Agent 'missing' not found"):
        await ExecuteAgentJob.to_spec(
            ExecuteAgentJobConfig(agent="missing", input="hello"),
            workspace="default",
            entity_client=entity_client,
            async_sdk=_sdk_with_files(),
            is_local=False,
        )


@pytest.mark.asyncio
async def test_to_spec_rejects_non_fabric_agent() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _agent(config_format="nat-workflow-v1")

    with pytest.raises(ValueError, match="only support 'nemo-agents-spec-v1'"):
        await ExecuteAgentJob.to_spec(
            ExecuteAgentJobConfig(agent="calc", input="hello"),
            workspace="default",
            entity_client=entity_client,
            async_sdk=_sdk_with_files(),
            is_local=False,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "error"),
    [
        ({}, "Agent 'default/calc' has no configuration"),
        ({"config_format": "nemo-agents-spec-v1", "name": "calc"}, "default_harness"),
    ],
)
async def test_to_spec_rejects_malformed_fabric_agent_config(config: dict, error: str) -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = Agent(
        name="calc",
        workspace="default",
        config=config,
        config_format="nemo-agents-spec-v1",
    )

    with pytest.raises(ValueError, match=error):
        await ExecuteAgentJob.to_spec(
            ExecuteAgentJobConfig(agent="calc", input="hello"),
            workspace="default",
            entity_client=entity_client,
            async_sdk=_sdk_with_files(),
            is_local=False,
        )


@pytest.mark.asyncio
async def test_to_spec_rejects_non_local_fabric_environment() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = Agent(
        name="calc",
        workspace="default",
        config=_agent_config(provider="docker"),
        config_format="nemo-agents-spec-v1",
    )

    with pytest.raises(ValueError, match="only support local Fabric environments"):
        await ExecuteAgentJob.to_spec(
            ExecuteAgentJobConfig(agent="calc", input="hello"),
            workspace="default",
            entity_client=entity_client,
            async_sdk=_sdk_with_files(),
            is_local=False,
        )


@pytest.mark.asyncio
@pytest.mark.usefixtures("installed_adapter_manifests")
async def test_to_spec_rejects_an_unknown_adapter_id_at_create_time() -> None:
    config = _agent_config()
    config["harnesses"]["hermes"]["kind"] = "nvidia.fabric.hermess"

    with pytest.raises(
        ValueError,
        match=r"Unsupported harness kind 'nvidia.fabric.hermess'.*accepted when the request names that image",
    ):
        await ExecuteAgentJob.to_spec(
            ExecuteAgentJobConfig(agent=AgentInline(config=config, config_format="nemo-agents-spec-v1"), input="hello"),
            workspace="default",
            entity_client=AsyncMock(),
            async_sdk=_sdk_with_files(),
            is_local=False,
        )


@pytest.mark.asyncio
@pytest.mark.usefixtures("installed_adapter_manifests")
async def test_to_spec_accepts_a_plugin_installed_adapter_id() -> None:
    config = _agent_config()
    config["harnesses"]["hermes"]["kind"] = "nvidia.helix.compass.analyst"

    spec = await ExecuteAgentJob.to_spec(
        ExecuteAgentJobConfig(agent=AgentInline(config=config, config_format="nemo-agents-spec-v1"), input="hello"),
        workspace="default",
        entity_client=AsyncMock(),
        async_sdk=_sdk_with_files(),
        is_local=False,
    )

    step_config = ExecuteAgentStepConfig.model_validate(spec)
    assert step_config.agent.config["harnesses"]["hermes"]["kind"] == "nvidia.helix.compass.analyst"


@pytest.mark.asyncio
@pytest.mark.usefixtures("installed_adapter_manifests")
async def test_to_spec_leaves_adapter_ids_to_a_requested_image() -> None:
    """The API can't see a custom image's adapters, so the job checks harness kinds inside that image."""
    config = _agent_config()
    config["harnesses"]["hermes"]["kind"] = "acme.custom.native"

    spec = await ExecuteAgentJob.to_spec(
        ExecuteAgentJobConfig(
            agent=AgentInline(config=config, config_format="nemo-agents-spec-v1"),
            input="hello",
            image="registry.example.com/acme/agent:1",
        ),
        workspace="default",
        entity_client=AsyncMock(),
        async_sdk=_sdk_with_files(),
        is_local=False,
    )

    step_config = ExecuteAgentStepConfig.model_validate(spec)
    assert step_config.agent.config["harnesses"]["hermes"]["kind"] == "acme.custom.native"


@pytest.mark.asyncio
async def test_to_spec_validates_and_canonicalizes_base_workdir() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _agent()
    sdk = _sdk_with_files()

    with patch("nemo_agents_plugin.jobs.execute.AsyncFilesClient.from_client", return_value=sdk.files):
        spec = await ExecuteAgentJob.to_spec(
            ExecuteAgentJobConfig(agent="calc", input="hello", workdir=AgentWorkdir(base_workdir="source#project")),
            workspace="default",
            entity_client=entity_client,
            async_sdk=sdk,
            is_local=False,
        )

    step_config = ExecuteAgentStepConfig.model_validate(spec)
    assert step_config.workdir is not None
    assert step_config.workdir.base_workdir == "default/source#project/"
    sdk.files.list_files.assert_awaited_once_with(
        workspace="default",
        name="source",
        query_params={"path": "project/"},
    )


@pytest.mark.asyncio
async def test_to_spec_rejects_a_bad_post_run_action_config_at_create_time() -> None:
    """The noop action takes no config, so this is a create-time 422, not a mid-job failure."""
    entity_client = AsyncMock()
    entity_client.get.return_value = _agent()

    with pytest.raises(ValueError, match="agents.noop"):
        await ExecuteAgentJob.to_spec(
            ExecuteAgentJobConfig(
                agent="calc",
                input="hello",
                post_run_action=PostRunActionConfig(kind="agents.noop", config={"agent": "research-agent"}),
            ),
            workspace="default",
            entity_client=entity_client,
            async_sdk=_sdk_with_files(),
            is_local=False,
        )


@pytest.mark.asyncio
async def test_to_spec_resolves_post_run_action() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _agent()

    with patch(
        "nemo_agents_plugin.jobs.execute.validate_post_run_action_config",
    ) as resolve:
        spec = await ExecuteAgentJob.to_spec(
            ExecuteAgentJobConfig(
                agent="calc",
                input="hello",
                post_run_action=PostRunActionConfig(kind="example.action", config={"raw": True}),
            ),
            workspace="default",
            entity_client=entity_client,
            async_sdk=_sdk_with_files(),
            is_local=False,
        )

    step_config = ExecuteAgentStepConfig.model_validate(spec)
    assert step_config.post_run_action == PostRunActionConfig(
        kind="example.action",
        config={"raw": True},
    )
    assert step_config.request.post_run_action == step_config.post_run_action
    resolve.assert_called_once_with("example.action", {"raw": True})


@pytest.mark.asyncio
async def test_to_spec_defaults_the_post_run_action_to_noop() -> None:
    spec = await ExecuteAgentJob.to_spec(
        ExecuteAgentJobConfig(
            agent=AgentInline(config=_agent_config(), config_format="nemo-agents-spec-v1"), input="hello"
        ),
        workspace="default",
        entity_client=AsyncMock(),
        async_sdk=_sdk_with_files(),
        is_local=False,
    )

    step_config = ExecuteAgentStepConfig.model_validate(spec)
    assert step_config.post_run_action == PostRunActionConfig(kind="agents.noop")


def test_job_config_rejects_the_removed_extension_field() -> None:
    """Ignoring it would silently run the job with no action at all."""
    with pytest.raises(ValueError, match="'extension' was renamed to 'post_run_action'"):
        ExecuteAgentJobConfig.model_validate(
            {"agent": "calc", "input": "hello", "extension": {"kind": "insights.analysis"}}
        )


@pytest.mark.asyncio
async def test_to_spec_rejects_the_unprefixed_noop_kind() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _agent()

    with pytest.raises(
        ValueError, match=r"Unknown post-run action 'noop'\. Installed post-run actions: .*agents\.noop"
    ):
        await ExecuteAgentJob.to_spec(
            ExecuteAgentJobConfig(agent="calc", input="hello", post_run_action=PostRunActionConfig(kind="noop")),
            workspace="default",
            entity_client=entity_client,
            async_sdk=_sdk_with_files(),
            is_local=False,
        )


@pytest.mark.asyncio
async def test_to_spec_rejects_single_file_base_workdir() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _agent()

    with pytest.raises(ValueError, match="non-empty directory"):
        sdk = _sdk_with_files(data=[])
        with patch("nemo_agents_plugin.jobs.execute.AsyncFilesClient.from_client", return_value=sdk.files):
            await ExecuteAgentJob.to_spec(
                ExecuteAgentJobConfig(
                    agent="calc",
                    input="hello",
                    workdir=AgentWorkdir(base_workdir="source#README.md"),
                ),
                workspace="default",
                entity_client=entity_client,
                async_sdk=sdk,
                is_local=False,
            )


def test_canonical_base_workdir_ref_rejects_path_escape() -> None:
    with pytest.raises(ValueError, match="path escapes"):
        _canonical_files_ref(
            "source#../secret",
            default_workspace="default",
            field="workdir.base_workdir",
            directory_like=True,
        )


def test_canonical_base_workdir_ref_rejects_legacy_path_shape() -> None:
    with pytest.raises(ValueError, match="before fileset paths"):
        _canonical_files_ref(
            "default/source/project",
            default_workspace="default",
            field="workdir.base_workdir",
            directory_like=True,
        )


def test_canonical_base_workdir_root_ref_keeps_fileset_delimiter() -> None:
    assert (
        _canonical_files_ref(
            "source",
            default_workspace="default",
            field="workdir.base_workdir",
            directory_like=True,
        )
        == "default/source#"
    )


@pytest.mark.parametrize("mount_path", ["../secret", "data/../../secret", "/abs/path", "", ".", "data\\file"])
def test_workdir_artifact_mount_rejects_escaping_mount_paths(mount_path: str) -> None:
    with pytest.raises(ValueError):
        AgentWorkdirArtifactMount(ref="source#artifact", mount_path=mount_path)


def test_workdir_artifact_mount_normalizes_relative_mount_path() -> None:
    mount = AgentWorkdirArtifactMount(ref="source#artifact", mount_path="data/output/")
    assert mount.mount_path == "data/output"


@pytest.mark.parametrize(
    ("left", "right"),
    [("data", "data"), ("data", "data/file.txt"), ("data/project", "data/project/subdir")],
)
def test_workdir_spec_rejects_overlapping_mount_paths(left: str, right: str) -> None:
    with pytest.raises(ValueError, match="overlapping mount paths"):
        AgentWorkdir(
            artifact_mounts=[
                AgentWorkdirArtifactMount(ref="source#a", mount_path=left),
                AgentWorkdirArtifactMount(ref="source#b", mount_path=right),
            ]
        )


def test_workdir_spec_allows_sibling_mount_paths() -> None:
    spec = AgentWorkdir(
        artifact_mounts=[
            AgentWorkdirArtifactMount(ref="source#a", mount_path="data/a"),
            AgentWorkdirArtifactMount(ref="source#b", mount_path="data/b"),
        ]
    )
    assert [mount.mount_path for mount in spec.artifact_mounts] == ["data/a", "data/b"]


@pytest.mark.asyncio
async def test_validate_agent_workdir_canonicalizes_refs() -> None:
    files_client = MagicMock()
    files_client.list_files = AsyncMock(return_value=_TypedFilesResponse([object()]))

    spec = await validate_agent_workdir(
        AgentWorkdir(
            base_workdir="source#project",
            artifact_mounts=[AgentWorkdirArtifactMount(ref="artifacts#notes.txt", mount_path="notes.txt")],
        ),
        files_client,
        default_workspace="default",
    )

    assert spec is not None
    assert spec.base_workdir == "default/source#project/"
    assert spec.artifact_mounts == [
        AgentWorkdirArtifactMount(ref="default/artifacts#notes.txt", mount_path="notes.txt")
    ]
    files_client.list_files.assert_has_awaits(
        [
            call(workspace="default", name="source", query_params={"path": "project/"}),
            call(workspace="default", name="artifacts", query_params={"path": "notes.txt"}),
        ]
    )


def test_materialize_agent_workdir_downloads_base_and_mounts(tmp_path: Path) -> None:
    files_client = MagicMock()
    calls: list[tuple[str, Path]] = []

    def _download(_files_client: object, ref: str, *, local_dir: Path) -> SimpleNamespace:
        del _files_client
        calls.append((ref, local_dir))
        local_dir.mkdir(parents=True, exist_ok=True)
        if ref == "default/source#project/":
            local = local_dir
            local.mkdir(parents=True, exist_ok=True)
            (local / "config.yaml").write_text("name: calc\n")
        else:
            local = local_dir / "notes.txt"
            local.write_text("mounted artifact\n")
        return SimpleNamespace(path=local, tmp_dir=local_dir)

    target = tmp_path / "workdir"

    with patch("nemo_agents_plugin.tasks.execute.workdir._download_fileset_ref", side_effect=_download):
        materialize_agent_workdir(
            AgentWorkdir(
                base_workdir="default/source#project/",
                artifact_mounts=[AgentWorkdirArtifactMount(ref="default/artifacts#notes.txt", mount_path="notes.txt")],
            ),
            files_client,
            target,
        )

    assert [ref for ref, _ in calls] == ["default/source#project/", "default/artifacts#notes.txt"]
    assert calls[0][1] == target
    assert (target / "config.yaml").read_text() == "name: calc\n"
    assert (target / "notes.txt").read_text() == "mounted artifact\n"


def test_materialize_agent_workdir_replaces_base_directory_for_directory_mount(tmp_path: Path) -> None:
    files_client = MagicMock()
    calls: list[tuple[str, Path]] = []

    def _download(_files_client: object, ref: str, *, local_dir: Path) -> SimpleNamespace:
        del _files_client
        calls.append((ref, local_dir))
        local_dir.mkdir(parents=True, exist_ok=True)
        if ref == "default/source#project/":
            local = local_dir
            (local / "data").mkdir(parents=True)
            (local / "data" / "stale.txt").write_text("stale\n")
        else:
            local = local_dir / "data"
            local.mkdir(parents=True)
            (local / "mounted.txt").write_text("mounted artifact\n")
        return SimpleNamespace(path=local, tmp_dir=local_dir)

    target = tmp_path / "workdir"

    with patch("nemo_agents_plugin.tasks.execute.workdir._download_fileset_ref", side_effect=_download):
        materialize_agent_workdir(
            AgentWorkdir(
                base_workdir="default/source#project/",
                artifact_mounts=[AgentWorkdirArtifactMount(ref="default/artifacts#data/", mount_path="data")],
            ),
            files_client,
            target,
        )

    assert [ref for ref, _ in calls] == ["default/source#project/", "default/artifacts#data/"]
    assert calls[0][1] == target
    assert not (target / "data" / "stale.txt").exists()
    assert (target / "data" / "mounted.txt").read_text() == "mounted artifact\n"


@pytest.mark.asyncio
async def test_compile_produces_single_cpu_step_with_canonical_config() -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
        workdir=AgentWorkdir(base_workdir="default/source#project/"),
    )

    with patch("nemo_agents_plugin.jobs.execute.AgentsConfig.get") as get_config:
        get_config.return_value.jobs.default_image = "registry.example/nhx-tasks:test"
        platform_spec = await ExecuteAgentJob.compile(
            workspace="default",
            spec=spec,
            entity_client=MagicMock(),
            job_name=None,
            async_sdk=MagicMock(),
        )

    steps = list(platform_spec["steps"])
    assert len(steps) == 1
    step = steps[0]
    assert step["name"] == "execute-agent"
    assert step["executor"]["provider"] == "cpu"
    assert step["executor"]["container"]["image"] == "registry.example/nhx-tasks:test"
    assert step["executor"]["container"]["command"] == ["nemo_agents_plugin.tasks.execute"]
    assert step["config"] == spec.model_dump(mode="json")
    step_config = cast(dict[str, Any], step["config"])
    assert cast(dict[str, Any], step_config["request"])["input"] == "hello"
    assert step["environment"] == []


async def _compiled_image(config: AgentsConfig, request: ExecuteAgentJobConfig) -> str | None:
    """Compile a minimal job under ``config`` and return the step's container image."""
    spec = ExecuteAgentStepConfig(request=request, agent=_resolved_agent())

    with patch("nemo_agents_plugin.jobs.execute.AgentsConfig.get", return_value=config):
        platform_spec = await ExecuteAgentJob.compile(
            workspace="default",
            spec=spec,
            entity_client=MagicMock(),
            job_name=None,
            async_sdk=MagicMock(),
        )

    steps = list(platform_spec["steps"])
    assert len(steps) == 1
    return cast(dict[str, Any], steps[0]["executor"]["container"]).get("image")


@pytest.mark.asyncio
async def test_request_image_wins_over_config_default() -> None:
    image = await _compiled_image(
        AgentsConfig(jobs=AgentJobsConfig(default_image="registry.example/config:test")),
        ExecuteAgentJobConfig(agent="calc", input="hello", image="registry.example/request:test"),
    )

    assert image == "registry.example/request:test"


@pytest.mark.asyncio
async def test_config_default_used_when_request_image_omitted() -> None:
    image = await _compiled_image(
        AgentsConfig(jobs=AgentJobsConfig(default_image="registry.example/config:test")),
        ExecuteAgentJobConfig(agent="calc", input="hello"),
    )

    assert image == "registry.example/config:test"


@pytest.mark.asyncio
async def test_compile_omits_image_to_inherit_substrate_chain() -> None:
    """No request image and no configured default leaves ``ContainerSpec.image`` unset.

    The job deliberately names no image of its own here: omitting it is what
    lets the jobs substrate apply the execution profile's ``default_task_image``
    and then the platform CPU tasks image, which the old ``nhx-api`` fallback
    short-circuited.
    """
    image = await _compiled_image(AgentsConfig(), ExecuteAgentJobConfig(agent="calc", input="hello"))

    assert image is None


@pytest.mark.asyncio
async def test_blank_config_default_is_treated_as_unset() -> None:
    """A whitespace-only configured default inherits the chain rather than reaching the runtime.

    Whitespace is truthy, so without stripping it would short-circuit the
    fallback and be handed to the container runtime as an unpullable image.
    ``ContainerSpec.image`` does not catch it either -- it rejects only the
    empty string. Config is not validated for this (no image setting anywhere
    in the platform is), so the resolution treats blank as absent.
    """
    blank_default = AgentsConfig(jobs=AgentJobsConfig(default_image="   "))

    assert await _compiled_image(blank_default, ExecuteAgentJobConfig(agent="calc", input="hello")) is None


@pytest.mark.asyncio
async def test_deployments_default_image_untouched_by_jobs_default() -> None:
    """The two knobs do not cross-talk in either direction."""
    deployments_only = AgentsConfig(deployments=DeploymentsRunnerConfig(default_image="registry.example/deploy:test"))
    assert await _compiled_image(deployments_only, ExecuteAgentJobConfig(agent="calc", input="hello")) is None

    jobs_only = AgentsConfig(jobs=AgentJobsConfig(default_image="registry.example/config:test"))
    assert jobs_only.deployments.default_image == ""


def test_blank_image_rejected() -> None:
    with pytest.raises(ValidationError, match="Image must not be blank"):
        ExecuteAgentJobConfig(agent="calc", input="hello", image="   ")


# --- environment / compute / secrets wiring ------------------------------------


def _env_ref_entity_client() -> AsyncMock:
    """Entity client that dereferences an AgentEnvironment ref chain.

    Dispatches ``entity_client.get`` by entity type: the Agent, then the
    AgentEnvironment and its EnvironmentSpec/ComputeSpec, all resolved by name.
    """
    entity_client = AsyncMock()

    env = AgentEnvironment(
        name="prod",
        workspace="default",
        environment_spec="default/prod-spec",
        compute_spec="default/prod-compute",
    )
    env_spec = AgentEnvironmentSpec(
        name="prod-spec",
        workspace="default",
        secrets={"OPENAI_API_KEY": "default/openai-key"},
    )
    compute_spec = AgentComputeSpec(
        name="prod-compute",
        workspace="default",
        resources=ComputeResources(limits={"cpu": "2", "memory": "4Gi", "nvidia.com/gpu": "1"}),
    )

    async def _get(entity_type: type, *, name: str, workspace: str) -> object:
        if entity_type is Agent:
            return _agent()
        if entity_type is AgentEnvironment:
            return env
        if entity_type is AgentEnvironmentSpec:
            return env_spec
        if entity_type is AgentComputeSpec:
            return compute_spec
        raise AssertionError(f"unexpected entity type {entity_type!r}")

    entity_client.get.side_effect = _get
    return entity_client


@pytest.mark.asyncio
async def test_to_spec_snapshots_resolved_environment_ref() -> None:
    entity_client = _env_ref_entity_client()

    spec = await ExecuteAgentJob.to_spec(
        ExecuteAgentJobConfig(agent="calc", input="hello", environment="default/prod"),
        workspace="default",
        entity_client=entity_client,
        async_sdk=_sdk_with_files(),
        is_local=False,
    )

    step_config = ExecuteAgentStepConfig.model_validate(spec)
    # Raw environment retained for provenance on the stored request.
    assert step_config.request.environment == "default/prod"
    # Secret refs collected from the EnvironmentSpec top-level secrets.
    assert step_config.secrets == {"OPENAI_API_KEY": "default/openai-key"}
    # Compute spec snapshotted (k8s-style resource maps preserved as-is).
    assert step_config.compute is not None
    assert step_config.compute.resources.limits == {"cpu": "2", "memory": "4Gi", "nvidia.com/gpu": "1"}


@pytest.mark.asyncio
async def test_to_spec_merges_inline_environment_into_config() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _agent()

    environment = AgentEnvironmentInline(
        environment_spec=EnvironmentSpecInline(
            env={"MY_FLAG": "on"},
            secrets={"OPENAI_API_KEY": "default/openai-key"},
        ),
        compute_spec=ComputeSpecInline(resources=ComputeResources(limits={"cpu": "500m"})),
    )

    spec = await ExecuteAgentJob.to_spec(
        ExecuteAgentJobConfig(agent="calc", input="hello", environment=environment),
        workspace="default",
        entity_client=entity_client,
        async_sdk=_sdk_with_files(),
        is_local=False,
    )

    step_config = ExecuteAgentStepConfig.model_validate(spec)
    # Inline env resolution never touches the entity store beyond the Agent.
    entity_client.get.assert_awaited_once_with(Agent, name="calc", workspace="default")
    # Plaintext env merged into the config's environment block.
    assert step_config.agent.config["environment"]["env"] == {"MY_FLAG": "on"}
    assert step_config.secrets == {"OPENAI_API_KEY": "default/openai-key"}
    assert step_config.compute is not None
    assert step_config.compute.resources.limits == {"cpu": "500m"}


@pytest.mark.asyncio
async def test_to_spec_mcp_secret_indirection_through_environment() -> None:
    """An Agent-declared MCP server is fulfilled with a secret ref via env-name
    indirection: the ref is collected into ``secrets`` (never written into the
    config), and the server's env references the value by name so the running
    step reads it from the process env the substrate populates.
    """
    mcp_agent_config: dict[str, Any] = {
        "config_format": "nemo-agents-spec-v1",
        "name": "calc",
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes"}},
        "models": {"default": {"provider": "openai", "model": "openai/gpt-5.4"}},
        "mcp": {"servers": {"search": {"transport": "streamable-http", "url": "http://agent-url"}}},
    }
    entity_client = AsyncMock()
    entity_client.get.return_value = Agent(
        name="calc",
        workspace="default",
        config=mcp_agent_config,
        config_format="nemo-agents-spec-v1",
    )

    environment = AgentEnvironmentInline(
        environment_spec=EnvironmentSpecInline(
            mcp={
                "search": McpFulfillment(
                    url="http://env-url",
                    env={"SEARCH_MODE": "fast"},
                    secrets={"SEARCH_TOKEN": "default/search-token"},
                ),
            },
        ),
    )

    spec = await ExecuteAgentJob.to_spec(
        ExecuteAgentJobConfig(agent="calc", input="hello", environment=environment),
        workspace="default",
        entity_client=entity_client,
        async_sdk=_sdk_with_files(),
        is_local=False,
    )

    step_config = ExecuteAgentStepConfig.model_validate(spec)
    server = step_config.agent.config["mcp"]["servers"]["search"]
    # Fulfillment url wins; non-secret env merged into the server config.
    assert server["url"] == "http://env-url"
    assert server["env"] == {"SEARCH_MODE": "fast"}
    # Secret ref collected for injection, NOT written into the server config.
    assert step_config.secrets == {"SEARCH_TOKEN": "default/search-token"}
    assert "SEARCH_TOKEN" not in server.get("env", {})


@pytest.mark.asyncio
async def test_to_spec_rejects_local_provider_credentials_before_workdir_access() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _agent()
    client = _sdk_with_files()
    environment = AgentEnvironmentInline(
        environment_spec=EnvironmentSpecInline(
            secrets={"SERVICE_SECRET": "default/service-token"},
            settings={
                "openshell": {
                    "providers": {
                        "service": {"profile": "service-read", "credentials": {"API_TOKEN": "SERVICE_SECRET"}}
                    }
                }
            },
        )
    )
    with pytest.raises(ValueError, match="require remote OpenShell"):
        await ExecuteAgentJob.to_spec(
            ExecuteAgentJobConfig(
                agent="calc",
                input="hello",
                environment=environment,
                workdir=AgentWorkdir(base_workdir="default/files"),
            ),
            workspace="default",
            entity_client=entity_client,
            async_sdk=client,
            is_local=True,
        )
    client.files.list_files.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "with_workdir,ethos_status", [(False, 200), (True, 200), (False, 404), (True, 404), (False, 403)]
)
async def test_provider_bindings_survive_remote_compilation_without_env_secrets(
    with_workdir: bool, ethos_status: int
) -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _agent()
    environment = AgentEnvironmentInline(
        environment_spec=EnvironmentSpecInline(
            secrets={"SERVICE_SECRET": "default/service-token"},
            settings={
                "openshell": {
                    "providers": {
                        "service": {"profile": "service-read", "credentials": {"API_TOKEN": "SERVICE_SECRET"}}
                    }
                }
            },
        )
    )
    client = _sdk_with_files()
    lookup = AsyncMock()
    if ethos_status != 200:
        error = NotFoundError if ethos_status == 404 else PermissionDeniedError
        lookup.side_effect = error(httpx.Response(ethos_status))
    with (
        patch("nemo_agents_plugin.jobs.execute.authorize_provider_references", new=AsyncMock()) as authorize,
        patch("nemo_agents_plugin.jobs.execute.AsyncFilesClient.get_fileset", new=lookup),
        pytest.raises(PermissionDeniedError) if ethos_status == 403 else nullcontext(),
    ):
        spec = await ExecuteAgentJob.to_spec(
            ExecuteAgentJobConfig(agent="calc", input="hello", environment=environment),
            workspace="default",
            entity_client=entity_client,
            async_sdk=client,
            is_local=False,
        )
    if ethos_status == 403:
        return
    authorize.assert_awaited_once()
    assert isinstance(spec, ExecuteAgentStepConfig)
    assert spec.secrets == {}
    assert spec.stage_ethos_fileset == (ethos_status == 200)
    lookup.assert_awaited_once_with(workspace=spec.agent.workspace, name=f"{spec.agent.name}-ethos")
    if with_workdir:
        spec.workdir = AgentWorkdir(base_workdir="other/source", artifact_mounts=[])
    with patch("nemo_agents_plugin.jobs.execute.AgentsConfig.get") as get_config:
        get_config.return_value.jobs.default_image = "registry.example/tasks:test"
        compiled = await ExecuteAgentJob.compile(
            workspace="default",
            spec=spec,
            entity_client=MagicMock(),
            job_name=None,
            async_sdk=MagicMock(),
            profile="sandbox",
        )
    step = compiled.steps[0]
    assert step.gateway_providers["service"].credentials == {"API_TOKEN": "default/service-token"}
    assert step.gateway_read_filesets == sorted(
        ([f"{spec.agent.workspace}/{spec.agent.name}-ethos"] if ethos_status == 200 else [])
        + (["other/source"] if with_workdir else [])
    )
    assert step.executor.profile == "sandbox"
    assert not step.environment
    assert "default/service-token" not in str(spec.agent.config)


@pytest.mark.asyncio
async def test_to_spec_with_no_environment_leaves_snapshot_empty() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _agent()

    spec = await ExecuteAgentJob.to_spec(
        ExecuteAgentJobConfig(agent="calc", input="hello"),
        workspace="default",
        entity_client=entity_client,
        async_sdk=_sdk_with_files(),
        is_local=False,
    )

    step_config = ExecuteAgentStepConfig.model_validate(spec)
    assert step_config.request.environment is None
    assert step_config.compute is None
    assert step_config.secrets == {}


@pytest.mark.asyncio
async def test_to_spec_rejects_missing_environment_ref() -> None:
    entity_client = AsyncMock()

    async def _get(entity_type: type, *, name: str, workspace: str) -> object:
        if entity_type is Agent:
            return _agent()
        raise NemoEntityNotFoundError("missing")

    entity_client.get.side_effect = _get

    with pytest.raises(ValueError, match="AgentEnvironment 'prod' not found"):
        await ExecuteAgentJob.to_spec(
            ExecuteAgentJobConfig(agent="calc", input="hello", environment="default/prod"),
            workspace="default",
            entity_client=entity_client,
            async_sdk=_sdk_with_files(),
            is_local=False,
        )


@pytest.mark.asyncio
async def test_to_spec_rejects_environment_spec_selecting_non_local_provider() -> None:
    entity_client = AsyncMock()
    entity_client.get.return_value = _agent()

    environment = AgentEnvironmentInline(
        environment_spec=EnvironmentSpecInline(provider="docker"),
    )

    with pytest.raises(ValueError, match="only support local Fabric environments"):
        await ExecuteAgentJob.to_spec(
            ExecuteAgentJobConfig(agent="calc", input="hello", environment=environment),
            workspace="default",
            entity_client=entity_client,
            async_sdk=_sdk_with_files(),
            is_local=False,
        )


@pytest.mark.asyncio
async def test_compile_injects_secret_env_and_compute_resources() -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
        compute=ComputeSpecInline(
            resources=ComputeResources(
                limits={"cpu": "2", "memory": "4Gi", "nvidia.com/gpu": "2"},
                requests={"cpu": "1", "memory": "2Gi"},
            )
        ),
        secrets={"OPENAI_API_KEY": "default/openai-key"},
    )

    with patch("nemo_agents_plugin.jobs.execute.AgentsConfig.get") as get_config:
        get_config.return_value.jobs.default_image = "registry.example/nhx-tasks:test"
        platform_spec = await ExecuteAgentJob.compile(
            workspace="default",
            spec=spec,
            entity_client=MagicMock(),
            job_name=None,
            async_sdk=MagicMock(),
        )

    step = list(platform_spec["steps"])[0]
    # Secret ref -> secret-backed env var.
    assert step["environment"] == [{"name": "OPENAI_API_KEY", "from_secret": {"name": "default/openai-key"}}]
    # Compute -> executor resources: cpu/memory pass through, gpu -> num_gpus.
    executor = cast(dict[str, Any], step["executor"])
    resources = executor["resources"]
    assert resources["limits"] == {"cpu": "2", "memory": "4Gi"}
    assert resources["requests"] == {"cpu": "1", "memory": "2Gi"}
    assert resources["num_gpus"] == 2


@pytest.mark.asyncio
async def test_compile_without_compute_omits_executor_resources() -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
    )

    with patch("nemo_agents_plugin.jobs.execute.AgentsConfig.get") as get_config:
        get_config.return_value.jobs.default_image = "registry.example/nhx-tasks:test"
        platform_spec = await ExecuteAgentJob.compile(
            workspace="default",
            spec=spec,
            entity_client=MagicMock(),
            job_name=None,
            async_sdk=MagicMock(),
        )

    step = list(platform_spec["steps"])[0]
    assert "resources" not in cast(dict[str, Any], step["executor"])
    assert step["environment"] == []


@pytest.mark.asyncio
async def test_compile_rejects_unsupported_compute_resource_key() -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
        compute=ComputeSpecInline(resources=ComputeResources(limits={"cpu": "1", "ephemeral-storage": "1Gi"})),
    )

    with (
        patch("nemo_agents_plugin.jobs.execute.AgentsConfig.get") as get_config,
        pytest.raises(HelixJobCompilationError, match="Unsupported compute resource key"),
    ):
        get_config.return_value.jobs.default_image = "registry.example/nhx-tasks:test"
        await ExecuteAgentJob.compile(
            workspace="default",
            spec=spec,
            entity_client=MagicMock(),
            job_name=None,
            async_sdk=MagicMock(),
        )


@pytest.mark.asyncio
async def test_compile_rejects_secret_env_colliding_with_reserved_name() -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
        secrets={"NHX_BASE_URL": "default/some-secret"},
    )

    with (
        patch("nemo_agents_plugin.jobs.execute.AgentsConfig.get") as get_config,
        pytest.raises(HelixJobCompilationError, match="reserved job env var name"),
    ):
        get_config.return_value.jobs.default_image = "registry.example/nhx-tasks:test"
        await ExecuteAgentJob.compile(
            workspace="default",
            spec=spec,
            entity_client=MagicMock(),
            job_name=None,
            async_sdk=MagicMock(),
        )


def test_run_without_input_workdir_saves_empty_input_snapshot(ctx: JobContext) -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
    )

    async def _invoke(request: Any) -> FabricRuntimeResult:
        assert request.input == "hello"
        assert request.base_dir == ctx.storage.ephemeral / "fabric"
        assert request.timeout_seconds == DEFAULT_AGENT_EXECUTION_TIMEOUT_SECONDS
        assert request.observe_progress is True
        (request.base_dir / "workspace" / "answer.txt").write_text("done\n")
        (request.base_dir / "artifacts" / "trace.json").write_text("{}\n")
        return FabricRuntimeResult(
            status="succeeded",
            response="done",
            metadata={"adapter_runner": "python"},
            runtime_id="runtime-1",
            invocation_id="invocation-1",
            request_id="request-1",
        )

    with patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke):
        result = ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert result["status"] == "completed"
    assert result["fabric_status"] == "succeeded"
    assert result["input_workdir"]["name"] == INPUT_WORKDIR_RESULT_NAME
    assert result["output_workdir"]["name"] == OUTPUT_WORKDIR_RESULT_NAME
    assert result["output_artifacts"]["name"] == OUTPUT_ARTIFACTS_RESULT_NAME
    assert result["fabric_run_result"]["name"] == FABRIC_RUN_RESULT_NAME
    assert list((ctx.storage.persistent / "results" / INPUT_WORKDIR_RESULT_NAME).iterdir()) == []
    assert (ctx.storage.persistent / "results" / OUTPUT_WORKDIR_RESULT_NAME / "answer.txt").read_text() == "done\n"
    assert (ctx.storage.persistent / "results" / OUTPUT_ARTIFACTS_RESULT_NAME / "trace.json").read_text() == "{}\n"
    payload = json.loads((ctx.storage.persistent / "results" / FABRIC_RUN_RESULT_NAME).read_text())
    assert payload["metadata"] == {"adapter_runner": "python"}
    assert payload["runtime_id"] == "runtime-1"


def test_run_reports_terminal_fabric_token_usage(ctx: JobContext) -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
    )

    async def _invoke(request: Any) -> FabricRuntimeResult:
        return FabricRuntimeResult(
            status="succeeded",
            output={"response": "done", "usage": {"input_tokens": 21, "output_tokens": 8}},
        )

    with patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert isinstance(ctx.usage, LocalJobUsageReporter)
    assert ctx.usage.latest is not None
    assert ctx.usage.latest.input_tokens == 21
    assert ctx.usage.latest.output_tokens == 8


def test_run_threads_custom_timeout_to_fabric(ctx: JobContext) -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello", timeout_seconds=12.5),
        agent=_resolved_agent(),
    )

    async def _invoke(request: Any) -> FabricRuntimeResult:
        assert request.timeout_seconds == 12.5
        return FabricRuntimeResult(status="succeeded")

    with patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke):
        result = ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert result["status"] == "completed"


def _patch_post_run_action(after_invoke: Callable[[Any], None] | None = None, *, resolved: list[str] | None = None):
    """Install a fake action for any kind, recording which kinds were resolved."""

    class _Action:
        def after_invoke(self, context: Any) -> None:
            if after_invoke is not None:
                after_invoke(context)

    def _resolve(kind: str) -> type[_Action]:
        if resolved is not None:
            resolved.append(kind)
        return _Action

    return patch("nemo_agents_plugin.jobs.execute.resolve_post_run_action", _resolve)


def test_run_invokes_post_run_action_on_success(ctx: JobContext) -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
        post_run_action=PostRunActionConfig(kind="example.action", config={"agent": "research-agent"}),
    )
    sdk = MagicMock()

    async def _invoke(request: Any) -> FabricRuntimeResult:
        return FabricRuntimeResult(status="succeeded", output={"answer": "done"})

    resolved: list[str] = []
    contexts: list[Any] = []

    def _after_invoke(context: Any) -> None:
        contexts.append(context)
        assert context.config == {"agent": "research-agent"}
        assert context.agent_name == "calc"
        assert context.agent_config == spec.agent.config
        assert context.fabric_result.output == {"answer": "done"}

    with (
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
        _patch_post_run_action(_after_invoke, resolved=resolved),
    ):
        result = ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx, sdk=sdk)

    assert result["status"] == "completed"
    assert "post_run_action" not in result
    assert resolved == ["example.action"]
    assert len(contexts) == 1


def test_run_hands_the_post_run_action_the_create_time_agent_config(ctx: JobContext) -> None:
    """Per-run wiring edits the step's config in place; the action must see neither
    that wiring (it points at a proxy that dies with the invocation) nor share the dict.
    """
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
        post_run_action=PostRunActionConfig(kind="example.action"),
    )
    created_config = json.loads(json.dumps(spec.agent.config))
    fabric_configs: list[dict[str, Any]] = []
    contexts: list[Any] = []

    async def _invoke(request: Any) -> FabricRuntimeResult:
        fabric_configs.append(request.agent_config.model_dump(mode="json"))
        return FabricRuntimeResult(status="succeeded", output={"answer": "done"})

    with (
        patch("nemo_agents_plugin.jobs.execute.wants_intake_atif_export", return_value=True),
        patch("nemo_agents_plugin.jobs.execute.supports_intake_atif_export", return_value=True),
        patch(
            "nemo_agents_plugin.jobs.execute.optional_platform_auth_proxy", lambda: nullcontext("http://127.0.0.1:9")
        ),
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
        _patch_post_run_action(contexts.append),
    ):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    # The real wiring did point the run at its proxy, so the test exercises the case it guards.
    assert "127.0.0.1:9" in json.dumps(fabric_configs[0])
    assert "127.0.0.1:9" not in json.dumps(contexts[0].agent_config)
    assert contexts[0].agent_config == created_config


def test_run_validates_the_post_run_action_config_against_the_image_before_invoking(ctx: JobContext) -> None:
    """Create time validated against the API server's version of the action; the task
    image's version may take a different config, which must not cost a full agent run.
    """

    class _ImageActionConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        destination: str

    class _ImageAction:
        config_model: ClassVar[type[BaseModel]] = _ImageActionConfig

        def after_invoke(self, context: Any) -> None:
            raise AssertionError("an action whose config was rejected must not run")

    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
        post_run_action=PostRunActionConfig(kind="example.action", config={"target": "accepted-by-the-server"}),
    )
    invoke = AsyncMock()

    with (
        patch("nemo_agents_plugin.jobs.execute.resolve_post_run_action", return_value=_ImageAction),
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", invoke),
        pytest.raises(ValueError, match=r"agent was not invoked: Invalid config for post-run action 'example\.action'"),
    ):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    invoke.assert_not_awaited()


def test_run_names_a_post_run_action_that_fails_to_import(ctx: JobContext) -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
        post_run_action=PostRunActionConfig(kind="broken.action"),
    )
    invoke = AsyncMock()

    with (
        patch(
            "nemo_agents_plugin.jobs.execute.resolve_post_run_action",
            side_effect=ImportError("No module named 'broken_plugin'"),
        ),
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", invoke),
        pytest.raises(ValueError, match=r"agent was not invoked: No module named 'broken_plugin'"),
    ):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    invoke.assert_not_awaited()


def test_run_skips_post_run_action_for_failed_fabric_result(ctx: JobContext) -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
        post_run_action=PostRunActionConfig(kind="example.action"),
    )

    async def _invoke(request: Any) -> FabricRuntimeResult:
        return FabricRuntimeResult(status="failed", error={"message": "adapter failed"})

    after_invoke = MagicMock()
    with (
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
        _patch_post_run_action(after_invoke),
    ):
        result = ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert result["status"] == "failed"
    after_invoke.assert_not_called()


def test_run_post_run_action_exception_is_logged(ctx: JobContext, caplog: pytest.LogCaptureFixture) -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
        post_run_action=PostRunActionConfig(kind="example.action"),
    )

    async def _invoke(request: Any) -> FabricRuntimeResult:
        return FabricRuntimeResult(status="succeeded", output={"answer": "done"})

    def _after_invoke(context: Any) -> None:
        raise RuntimeError("action exploded")

    with (
        caplog.at_level(logging.ERROR, logger="nemo_agents_plugin.jobs.execute"),
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
        _patch_post_run_action(_after_invoke),
        pytest.raises(RuntimeError, match="action exploded"),
    ):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert "Post-run action 'example.action' failed." in caplog.text


def test_run_fails_before_invoking_the_agent_when_its_post_run_action_is_not_installed(ctx: JobContext) -> None:
    """The task image can lack a plugin the API server had at create time, and a
    stored job can outlive its plugin; neither should cost a full agent run.
    """
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
        post_run_action=PostRunActionConfig(kind="uninstalled.action"),
    )
    invoke = AsyncMock()

    with (
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", invoke),
        pytest.raises(ValueError, match=r"agent was not invoked: Unknown post-run action 'uninstalled\.action'"),
    ):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    invoke.assert_not_awaited()


def test_run_rejects_non_fabric_step_config(ctx: JobContext) -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=ResolvedAgentConfig(
            name="calc",
            workspace="default",
            config={"workflow": {}},
            config_format="nat-workflow-v1",
        ),
    )

    with pytest.raises(ValueError, match="only support 'nemo-agents-spec-v1'"):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)


def test_run_downloads_and_registers_input_workdir(ctx: JobContext) -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(
            agent="calc",
            input="hello",
            workdir=AgentWorkdir(base_workdir="source#project/"),
        ),
        agent=_resolved_agent(),
        workdir=AgentWorkdir(
            base_workdir="default/source#project/",
            artifact_mounts=[AgentWorkdirArtifactMount(ref="default/artifacts#notes.txt", mount_path="notes.txt")],
        ),
    )
    sdk = MagicMock()
    download_calls: list[tuple[str, str]] = []

    def _download(_files_client: object, ref: str, *, local_dir: Path) -> SimpleNamespace:
        del _files_client
        download_calls.append((ref, str(local_dir)))
        local_dir.mkdir(parents=True, exist_ok=True)
        if ref == "default/source#project/":
            local = local_dir
            local.mkdir(parents=True, exist_ok=True)
            (local / "config.yaml").write_text("name: calc\n")
        else:
            local = local_dir / "notes.txt"
            local.write_text("mounted artifact\n")
        return SimpleNamespace(path=local, tmp_dir=local_dir)

    async def _invoke(request: Any) -> FabricRuntimeResult:
        (request.base_dir / "workspace" / "answer.txt").write_text("done\n")
        (request.base_dir / "artifacts" / "artifact.txt").write_text("artifact\n")
        return FabricRuntimeResult(status="succeeded", response="done")

    with (
        patch("nemo_agents_plugin.jobs.execute.FilesClient.from_client", return_value=sdk.files),
        patch("nemo_agents_plugin.tasks.execute.workdir._download_fileset_ref", side_effect=_download),
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
    ):
        result = ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx, sdk=sdk)

    assert result["status"] == "completed"
    assert result["input_workdir"]["name"] == "input_workdir"
    assert [remote_path for remote_path, _ in download_calls] == [
        "default/source#project/",
        "default/artifacts#notes.txt",
    ]
    assert (ctx.storage.persistent / "results" / "input_workdir" / "config.yaml").read_text() == "name: calc\n"
    assert (ctx.storage.persistent / "results" / "input_workdir" / "notes.txt").read_text() == "mounted artifact\n"
    assert (ctx.storage.persistent / "results" / "output_workdir" / "answer.txt").read_text() == "done\n"
    assert (ctx.storage.persistent / "results" / "output_artifacts" / "artifact.txt").read_text() == "artifact\n"


def test_run_clears_stale_input_workdir_before_materializing(ctx: JobContext) -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(
            agent="calc",
            input="hello",
            workdir=AgentWorkdir(base_workdir="source#project/"),
        ),
        agent=_resolved_agent(),
        workdir=AgentWorkdir(base_workdir="default/source#project/"),
    )
    stale_workdir = ctx.storage.ephemeral / "fabric" / "workspace"
    stale_workdir.mkdir(parents=True)
    (stale_workdir / "stale.txt").write_text("stale\n")
    sdk = MagicMock()

    def _download(_files_client: object, ref: str, *, local_dir: Path) -> SimpleNamespace:
        del _files_client
        assert ref == "default/source#project/"
        local = local_dir
        assert not (local / "stale.txt").exists()
        local.mkdir(parents=True, exist_ok=True)
        (local / "config.yaml").write_text("name: calc\n")
        return SimpleNamespace(path=local, tmp_dir=local_dir)

    async def _invoke(request: Any) -> FabricRuntimeResult:
        return FabricRuntimeResult(status="succeeded")

    with (
        patch("nemo_agents_plugin.jobs.execute.FilesClient.from_client", return_value=sdk.files),
        patch("nemo_agents_plugin.tasks.execute.workdir._download_fileset_ref", side_effect=_download),
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
    ):
        result = ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx, sdk=sdk)

    assert result["status"] == "completed"
    assert result["input_workdir"]["name"] == "input_workdir"
    saved_workdir = ctx.storage.persistent / "results" / "input_workdir"
    assert not (saved_workdir / "stale.txt").exists()
    assert (saved_workdir / "config.yaml").read_text() == "name: calc\n"


def test_run_clears_stale_artifacts_dir_before_invoking(ctx: JobContext) -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
    )
    stale_artifacts = ctx.storage.ephemeral / "fabric" / "artifacts"
    stale_artifacts.mkdir(parents=True)
    (stale_artifacts / "stale.txt").write_text("stale\n")

    async def _invoke(request: Any) -> FabricRuntimeResult:
        assert not (request.base_dir / "artifacts" / "stale.txt").exists()
        (request.base_dir / "artifacts" / "fresh.txt").write_text("fresh\n")
        return FabricRuntimeResult(status="succeeded")

    with patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke):
        result = ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert result["status"] == "completed"
    saved_artifacts = ctx.storage.persistent / "results" / "output_artifacts"
    assert not (saved_artifacts / "stale.txt").exists()
    assert (saved_artifacts / "fresh.txt").read_text() == "fresh\n"


def test_run_failed_download_does_not_register_partial_result(ctx: JobContext) -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(
            agent="calc",
            input="hello",
            workdir=AgentWorkdir(base_workdir="source#project/"),
        ),
        agent=_resolved_agent(),
        workdir=AgentWorkdir(base_workdir="default/source#project/"),
    )
    sdk = MagicMock()

    with (
        patch("nemo_agents_plugin.jobs.execute.FilesClient.from_client", return_value=sdk.files),
        patch(
            "nemo_agents_plugin.tasks.execute.workdir._download_fileset_ref",
            side_effect=RuntimeError("download failed"),
        ),
        pytest.raises(RuntimeError, match="download failed"),
    ):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx, sdk=sdk)

    assert not (ctx.storage.persistent / "results" / "input_workdir").exists()


def test_run_failed_fabric_result_saves_results_and_marks_job_failed(ctx: JobContext) -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
    )

    async def _invoke(request: Any) -> FabricRuntimeResult:
        (request.base_dir / "workspace" / "partial.txt").write_text("partial\n")
        return FabricRuntimeResult(
            status="failed",
            error={"stage": "invoke", "message": "adapter failed"},
        )

    with patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke):
        result = ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert result["status"] == "failed"
    assert result["fabric_status"] == "failed"
    assert (ctx.storage.persistent / "results" / OUTPUT_WORKDIR_RESULT_NAME / "partial.txt").read_text() == "partial\n"
    payload = json.loads((ctx.storage.persistent / "results" / FABRIC_RUN_RESULT_NAME).read_text())
    assert payload["status"] == "failed"
    assert payload["error"] == {"stage": "invoke", "message": "adapter failed"}
    assert not (ctx.storage.persistent / "results" / FABRIC_ERROR_RESULT_NAME).exists()


def test_run_fabric_exception_saves_best_effort_error_results(ctx: JobContext) -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
    )

    async def _invoke(request: Any) -> FabricRuntimeResult:
        (request.base_dir / "workspace" / "partial.txt").write_text("partial\n")
        raise RuntimeError("fabric exploded")

    with (
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
        pytest.raises(RuntimeError, match="fabric exploded"),
    ):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert (ctx.storage.persistent / "results" / INPUT_WORKDIR_RESULT_NAME).is_dir()
    assert (ctx.storage.persistent / "results" / OUTPUT_WORKDIR_RESULT_NAME / "partial.txt").read_text() == "partial\n"
    assert (ctx.storage.persistent / "results" / OUTPUT_ARTIFACTS_RESULT_NAME).is_dir()
    payload = json.loads((ctx.storage.persistent / "results" / FABRIC_ERROR_RESULT_NAME).read_text())
    assert payload["type"] == "RuntimeError"
    assert payload["message"] == "fabric exploded"
    assert payload["fabric_status"] == "error"


def test_run_fabric_exception_preserves_original_error_if_result_save_fails(ctx: JobContext) -> None:
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
    )
    original_save = ctx.results.save

    async def _invoke(request: Any) -> FabricRuntimeResult:
        (request.base_dir / "workspace" / "partial.txt").write_text("partial\n")
        raise RuntimeError("fabric exploded")

    def _save(name: str, local_path: str | Path, **kwargs: Any) -> Any:
        if name == OUTPUT_WORKDIR_RESULT_NAME:
            raise RuntimeError("save failed")
        return original_save(name, local_path, **kwargs)

    cast(Any, ctx.results).save = _save

    with (
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
        pytest.raises(RuntimeError, match="fabric exploded"),
    ):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert (ctx.storage.persistent / "results" / FABRIC_ERROR_RESULT_NAME).exists()


def test_execute_job_create_route_stores_canonical_step_config() -> None:
    app = FastAPI()
    app.include_router(add_job_routes(ExecuteAgentJob), prefix="/apis/agents/v2/workspaces/{workspace}")

    entity_client = AsyncMock()
    entity_client.get.return_value = _agent()
    sdk = _sdk_with_files()
    app.dependency_overrides[get_entity_client] = lambda: entity_client
    app.dependency_overrides[get_nemo_client] = lambda: sdk

    captured_body: dict[str, Any] = {}

    async def _create_job(*, workspace: str, body: object) -> MagicMock:
        del workspace
        typed_body = cast(Any, body)
        captured_body["body"] = body
        response = MagicMock()
        response.data.return_value = SimpleNamespace(
            id="job-1",
            attempt_id="attempt-1",
            name="execute-1",
            description=None,
            project=None,
            workspace="default",
            created_at=None,
            updated_at=None,
            spec=typed_body.spec,
            status="created",
            status_details=None,
            error_details=None,
            ownership=None,
            custom_fields=None,
            control=None,
        )
        return response

    fake_jobs = SimpleNamespace(create_job=_create_job)
    with (
        patch("nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client", return_value=fake_jobs),
        patch("nemo_agents_plugin.jobs.execute.AsyncFilesClient.from_client", return_value=sdk.files),
    ):
        response = TestClient(app).post(
            "/apis/agents/v2/workspaces/default/jobs/execute",
            json={
                "name": "execute-1",
                "spec": {"agent": "calc", "input": "hello", "workdir": {"base_workdir": "source#project"}},
            },
        )

    assert response.status_code == 201, response.text
    body = captured_body["body"]
    assert body.source == "nemo-agents-plugin"
    assert body.spec["agent"]["name"] == "calc"
    assert body.spec["agent"]["workspace"] == "default"
    assert body.spec["agent"]["config_format"] == "nemo-agents-spec-v1"
    # The stored job spec carries the qualified ref that ``filter[spec.agent_ref]`` matches.
    assert body.spec["agent_ref"] == "default/calc"
    assert body.spec["agent"]["config"]["environment"] == {
        "provider": "local",
        "workspace": "./workspace",
        "artifacts": "./artifacts",
        "connection": {},
        "env": {},
        "metadata": {},
        "settings": {},
    }
    assert (
        body.spec["agent"]["config"]["models"]["default"]["base_url"]
        == "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    )
    assert body.spec["request"] == {
        "agent": "calc",
        "input": "hello",
        "context": {},
        "environment": None,
        "workdir": {"base_workdir": "source#project", "artifact_mounts": []},
        "timeout_seconds": DEFAULT_AGENT_EXECUTION_TIMEOUT_SECONDS,
        "auto_telemetry": True,
        "image": "",
        "post_run_action": None,
    }
    assert body.spec["workdir"] == {"base_workdir": "default/source#project/", "artifact_mounts": []}
    assert body.platform_spec.steps[0].name == "execute-agent"
    assert response.json()["spec"]["workdir"]["base_workdir"] == "default/source#project/"


def test_execute_job_create_route_maps_reserved_secret_env_to_422() -> None:
    """A reserved-name secret-env collision must surface as a 422, not a 500.

    The collision is detected in ``compile`` (``_secret_environment``). The jobs
    create route only translates ``HelixJobCompilationError`` into a 422 - a
    bare ``ValueError`` escaping ``compile`` would fall through to the global
    handler as an opaque 500. This guards that the error reaches the client as a
    descriptive 422 at the HTTP boundary (the unit test on ``_secret_environment``
    only checks the raised exception, not the mapped status code).
    """
    app = FastAPI()
    app.include_router(add_job_routes(ExecuteAgentJob), prefix="/apis/agents/v2/workspaces/{workspace}")

    # Agent resolves, plus an environment whose EnvironmentSpec maps a secret env
    # var onto a reserved job env var name (NHX_BASE_URL).
    env = AgentEnvironment(name="prod", workspace="default", environment_spec="default/prod-spec")
    env_spec = AgentEnvironmentSpec(
        name="prod-spec", workspace="default", secrets={"NHX_BASE_URL": "default/some-secret"}
    )

    async def _get(entity_type: type, *, name: str, workspace: str) -> object:
        if entity_type is Agent:
            return _agent()
        if entity_type is AgentEnvironment:
            return env
        if entity_type is AgentEnvironmentSpec:
            return env_spec
        raise AssertionError(f"unexpected entity type {entity_type!r}")

    entity_client = AsyncMock()
    entity_client.get.side_effect = _get
    app.dependency_overrides[get_entity_client] = lambda: entity_client
    app.dependency_overrides[get_nemo_client] = lambda: _sdk_with_files()

    with patch("nemo_agents_plugin.jobs.execute.AgentsConfig.get") as get_config:
        get_config.return_value.jobs.default_image = "registry.example/nhx-tasks:test"
        response = TestClient(app, raise_server_exceptions=False).post(
            "/apis/agents/v2/workspaces/default/jobs/execute",
            json={"name": "execute-1", "spec": {"agent": "calc", "input": "hello", "environment": "default/prod"}},
        )

    assert response.status_code == 422, response.text
    assert "reserved job env var name" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Inline agent definitions
# ---------------------------------------------------------------------------


def test_job_config_accepts_an_inline_agent_definition() -> None:
    config = ExecuteAgentJobConfig(
        agent=AgentInline(config=_agent_config(), config_format="nemo-agents-spec-v1"), input="hello"
    )

    assert isinstance(config.agent, AgentInline)
    assert config.agent.config_format == "nemo-agents-spec-v1"


def test_job_config_still_rejects_a_malformed_agent_ref() -> None:
    """The ref pattern moved into a validator; the string arm must keep enforcing it."""
    with pytest.raises(ValueError, match="must be an entity name or a 'workspace/name' ref"):
        ExecuteAgentJobConfig(agent="not a valid ref!", input="hello")


@pytest.mark.asyncio
async def test_to_spec_resolves_an_inline_agent_without_touching_the_entity_store() -> None:
    entity_client = AsyncMock()

    spec = await ExecuteAgentJob.to_spec(
        ExecuteAgentJobConfig(
            agent=AgentInline(config=_agent_config(), config_format="nemo-agents-spec-v1"), input="hello"
        ),
        workspace="research",
        entity_client=entity_client,
        async_sdk=_sdk_with_files(),
        is_local=False,
    )

    step_config = ExecuteAgentStepConfig.model_validate(spec)
    # Name comes from the config; the run is attributed to the job's workspace.
    assert step_config.agent.name == "calc"
    assert step_config.agent.workspace == "research"
    assert step_config.agent.config_format == "nemo-agents-spec-v1"
    # Not an entity, so the job belongs to no agent's runs.
    assert step_config.agent_ref is None
    entity_client.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_to_spec_snapshots_the_inline_config_like_an_entity_backed_one() -> None:
    """Provenance parity: the resolved config is stored on the step either way."""
    spec = await ExecuteAgentJob.to_spec(
        ExecuteAgentJobConfig(
            agent=AgentInline(config=_agent_config(), config_format="nemo-agents-spec-v1"), input="hello"
        ),
        workspace="default",
        entity_client=AsyncMock(),
        async_sdk=_sdk_with_files(),
        is_local=False,
    )

    step_config = ExecuteAgentStepConfig.model_validate(spec)
    assert (
        step_config.agent.config["models"]["default"]["base_url"]
        == "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    )


@pytest.mark.asyncio
async def test_to_spec_rejects_a_malformed_inline_agent_config() -> None:
    with pytest.raises(ValueError, match="Inline agent config is invalid"):
        await ExecuteAgentJob.to_spec(
            ExecuteAgentJobConfig(
                agent=AgentInline(config={"config_format": "nemo-agents-spec-v1"}, config_format="nemo-agents-spec-v1"),
                input="x",
            ),
            workspace="default",
            entity_client=AsyncMock(),
            async_sdk=_sdk_with_files(),
            is_local=False,
        )


@pytest.mark.asyncio
async def test_to_spec_rejects_a_non_fabric_inline_config_format() -> None:
    with pytest.raises(ValueError, match="is not supported"):
        await ExecuteAgentJob.to_spec(
            ExecuteAgentJobConfig(
                agent=AgentInline(config={"workflow": {}}, config_format="nat-workflow-v1"), input="x"
            ),
            workspace="default",
            entity_client=AsyncMock(),
            async_sdk=_sdk_with_files(),
            is_local=False,
        )


@pytest.mark.asyncio
async def test_to_spec_merges_an_environment_into_an_inline_agent() -> None:
    """Inline configs go through the same environment merge as entity-backed ones."""
    spec = await ExecuteAgentJob.to_spec(
        ExecuteAgentJobConfig(
            agent=AgentInline(config=_agent_config(), config_format="nemo-agents-spec-v1"),
            input="hello",
            environment=AgentEnvironmentInline(
                environment_spec=EnvironmentSpecInline(env={"FEATURE_FLAG": "on"}),
            ),
        ),
        workspace="default",
        entity_client=AsyncMock(),
        async_sdk=_sdk_with_files(),
        is_local=False,
    )

    step_config = ExecuteAgentStepConfig.model_validate(spec)
    assert step_config.agent.config["environment"]["env"]["FEATURE_FLAG"] == "on"


def test_agent_inline_matches_the_agent_entity_shape() -> None:
    """AgentInline is the Agent entity's fields without identity, so Agent inherits it."""
    assert issubclass(Agent, AgentInline)
    for field, info in AgentInline.model_fields.items():
        entity_field = Agent.model_fields[field]
        assert entity_field.is_required() == info.is_required(), field
        assert entity_field.default == info.default, field


def test_inline_agent_rejects_a_non_fabric_config_format_at_request_time() -> None:
    """The narrowing lives on the job, not on the shared AgentInline model."""
    with pytest.raises(ValueError, match="only support 'nemo-agents-spec-v1'"):
        ExecuteAgentJobConfig(agent=AgentInline(config={"workflow": {}}), input="hello")


def test_inline_agent_rejects_an_empty_config() -> None:
    with pytest.raises(ValueError, match="require a non-empty config"):
        ExecuteAgentJobConfig(agent=AgentInline(config={}, config_format="nemo-agents-spec-v1"), input="hello")


def test_agent_entity_may_still_hold_a_nat_workflow_config() -> None:
    """AgentInline stays permissive so the entity keeps its legacy default."""
    entity = Agent(name="legacy", workspace="default", config={"workflow": {}})

    assert entity.config_format == "nat-workflow-v1"


def test_run_failed_fabric_result_reports_the_reason_in_the_log(
    ctx: JobContext, caplog: pytest.LogCaptureFixture
) -> None:
    """A Fabric result that reports failure never raises, so this is the only
    place the reason is stated. Without it the step exits non-zero having
    logged nothing at all."""
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
    )

    async def _invoke(request: Any) -> FabricRuntimeResult:
        return FabricRuntimeResult(
            status="failed",
            error={"stage": "invoke", "code": "insights_analyst_failed", "message": "LLM API error"},
            runtime_id="runtime-1",
            invocation_id="invocation-1",
        )

    with (
        caplog.at_level(logging.ERROR, logger="nemo_agents_plugin.jobs.execute"),
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
    ):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    logged = caplog.text
    assert "LLM API error" in logged
    assert "insights_analyst_failed" in logged
    assert "runtime-1" in logged
    assert "invocation-1" in logged


def test_run_logs_the_agent_stderr_when_the_run_fails(ctx: JobContext, caplog: pytest.LogCaptureFixture) -> None:
    """Fabric captures the agent process's stderr to a file, so on failure the
    real traceback reaches the container's own logs nowhere else."""
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
    )

    async def _invoke(request: Any) -> FabricRuntimeResult:
        # Mirrors the layout Fabric creates: a runtime/invocation directory it
        # names itself, under the artifacts dir.
        invocation_dir = request.base_dir / "artifacts" / ".fabric" / "runtime-1" / "invocation-1"
        invocation_dir.mkdir(parents=True, exist_ok=True)
        (invocation_dir / "stderr.txt").write_text("Traceback (most recent call last):\nValueError: the actual cause\n")
        return FabricRuntimeResult(status="failed", error={"message": "agent failed"})

    with (
        caplog.at_level(logging.ERROR, logger="nemo_agents_plugin.jobs.execute"),
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
    ):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert "ValueError: the actual cause" in caplog.text


def test_run_logs_the_agent_stderr_when_fabric_raises(ctx: JobContext, caplog: pytest.LogCaptureFixture) -> None:
    """Fabric produced no result, so the agent's stderr is the only account of
    how far it got - which is the case telemetry cannot cover."""
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
    )

    async def _invoke(request: Any) -> FabricRuntimeResult:
        invocation_dir = request.base_dir / "artifacts" / ".fabric" / "runtime-1" / "invocation-1"
        invocation_dir.mkdir(parents=True, exist_ok=True)
        (invocation_dir / "stderr.txt").write_text("got as far as loading traces\n")
        raise TimeoutError("fabric timed out")

    with (
        caplog.at_level(logging.ERROR, logger="nemo_agents_plugin.jobs.execute"),
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
        pytest.raises(TimeoutError),
    ):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert "got as far as loading traces" in caplog.text


def test_run_does_not_log_agent_stderr_on_success(ctx: JobContext, caplog: pytest.LogCaptureFixture) -> None:
    """A healthy run's execution record belongs in telemetry, which carries it
    live and structured; duplicating it into the job log is noise."""
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
    )

    async def _invoke(request: Any) -> FabricRuntimeResult:
        invocation_dir = request.base_dir / "artifacts" / ".fabric" / "runtime-1" / "invocation-1"
        invocation_dir.mkdir(parents=True, exist_ok=True)
        (invocation_dir / "stderr.txt").write_text("routine chatter nobody needs\n")
        return FabricRuntimeResult(status="succeeded", response="done")

    with (
        caplog.at_level(logging.INFO, logger="nemo_agents_plugin.jobs.execute"),
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
    ):
        result = ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert result["status"] == "completed"
    assert "routine chatter nobody needs" not in caplog.text


def test_run_ignores_the_agent_stdout_file(ctx: JobContext, caplog: pytest.LogCaptureFixture) -> None:
    """``stdout.txt`` is the adapter's protocol channel - a single JSON response
    already saved as ``fabric_run_result`` - not a log stream."""
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
    )

    async def _invoke(request: Any) -> FabricRuntimeResult:
        invocation_dir = request.base_dir / "artifacts" / ".fabric" / "runtime-1" / "invocation-1"
        invocation_dir.mkdir(parents=True, exist_ok=True)
        (invocation_dir / "stdout.txt").write_text('{"response":"the protocol payload"}\n')
        (invocation_dir / "stderr.txt").write_text("the real diagnostics\n")
        return FabricRuntimeResult(status="failed", error={"message": "agent failed"})

    with (
        caplog.at_level(logging.ERROR, logger="nemo_agents_plugin.jobs.execute"),
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
    ):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert "the real diagnostics" in caplog.text
    assert "the protocol payload" not in caplog.text


def test_run_logs_the_whole_agent_stderr_for_realistic_output(
    ctx: JobContext, caplog: pytest.LogCaptureFixture
) -> None:
    """The byte cap exists for the pathological case only; ordinary output,
    however long, must arrive complete."""
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
    )
    long_stderr = "\n".join(f"line {n}" for n in range(5000))

    async def _invoke(request: Any) -> FabricRuntimeResult:
        invocation_dir = request.base_dir / "artifacts" / ".fabric" / "runtime-1" / "invocation-1"
        invocation_dir.mkdir(parents=True, exist_ok=True)
        (invocation_dir / "stderr.txt").write_text(long_stderr)
        return FabricRuntimeResult(status="failed", error={"message": "agent failed"})

    with (
        caplog.at_level(logging.ERROR, logger="nemo_agents_plugin.jobs.execute"),
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
    ):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert "line 0" in caplog.text
    assert "line 4999" in caplog.text


def test_run_tolerates_a_failure_with_no_agent_stderr(ctx: JobContext) -> None:
    """The dump runs on an already-failing path; it must not replace the error."""
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
    )

    async def _invoke(request: Any) -> FabricRuntimeResult:
        return FabricRuntimeResult(status="failed", error={"message": "agent failed"})

    with patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke):
        result = ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert result["status"] == "failed"


def test_log_agent_stderr_reports_an_empty_stream(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """An empty stderr is itself a finding: the agent logged nothing at all."""
    (tmp_path / "stderr.txt").write_text("   \n")

    with caplog.at_level(logging.WARNING, logger="nemo_agents_plugin.jobs.execute"):
        _log_agent_stderr(tmp_path)

    assert "was empty" in caplog.text


def test_log_agent_stderr_survives_a_missing_artifacts_directory() -> None:
    """Never the reason a failing run fails differently."""
    _log_agent_stderr(Path("/nonexistent/artifacts"))


def test_run_logs_progress_through_a_successful_run(ctx: JobContext, caplog: pytest.LogCaptureFixture) -> None:
    """A healthy run should account for itself: which agent, that it was
    invoked, and what came back."""
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello", timeout_seconds=300),
        agent=_resolved_agent(),
    )

    async def _invoke(request: Any) -> FabricRuntimeResult:
        return FabricRuntimeResult(status="succeeded", response="done")

    with (
        caplog.at_level(logging.INFO, logger="nemo_agents_plugin.jobs.execute"),
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
    ):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert "Executing agent default/calc (timeout 300s)." in caplog.text
    assert "Invoking agent default/calc." in caplog.text
    assert "returned status=succeeded after" in caplog.text


def test_run_names_the_agent_before_validating_its_config(ctx: JobContext, caplog: pytest.LogCaptureFixture) -> None:
    """A config that fails to parse must still say which agent it belonged to."""
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=ResolvedAgentConfig(
            name="calc",
            workspace="default",
            config={"not": "a valid agent config"},
            config_format="nemo-agents-spec-v1",
        ),
    )

    with (
        caplog.at_level(logging.INFO, logger="nemo_agents_plugin.jobs.execute"),
        pytest.raises(Exception),
    ):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert "Executing agent default/calc" in caplog.text


def test_run_does_not_announce_workdir_staging_when_there_is_none(
    ctx: JobContext, caplog: pytest.LogCaptureFixture
) -> None:
    """Most runs stage nothing; the line would be noise."""
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
    )

    async def _invoke(request: Any) -> FabricRuntimeResult:
        return FabricRuntimeResult(status="succeeded", response="done")

    with (
        caplog.at_level(logging.INFO, logger="nemo_agents_plugin.jobs.execute"),
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
    ):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert "Staging workdir" not in caplog.text


def test_run_logs_elapsed_time_when_fabric_raises(ctx: JobContext, caplog: pytest.LogCaptureFixture) -> None:
    """How long a run got before dying is part of diagnosing a timeout."""
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_resolved_agent(),
    )

    async def _invoke(request: Any) -> FabricRuntimeResult:
        raise TimeoutError("fabric timed out")

    with (
        caplog.at_level(logging.INFO, logger="nemo_agents_plugin.jobs.execute"),
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
        pytest.raises(TimeoutError),
    ):
        ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx)

    assert "Fabric invocation failed for agent default/calc after" in caplog.text


def test_log_agent_stderr_tails_a_stream_over_the_cap(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The artifacts dir is agent-writable, so an unbounded read here could
    exhaust the task's memory while it reports a failure - destroying the very
    diagnostics this produces."""
    monkeypatch.setattr("nemo_agents_plugin.jobs.execute._MAX_LOGGED_STDERR_BYTES", 200)
    (tmp_path / "stderr.txt").write_text("A" * 5000 + "\nthe failure is at the end\n")

    with caplog.at_level(logging.ERROR, logger="nemo_agents_plugin.jobs.execute"):
        _log_agent_stderr(tmp_path)

    assert "the failure is at the end" in caplog.text, "the tail is what matters for a long run"
    assert "earlier bytes omitted" in caplog.text
    assert "A" * 1000 not in caplog.text, "the whole file was read despite the cap"


def test_log_agent_stderr_caps_the_number_of_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent can write as many of these as it likes into its artifacts dir."""
    monkeypatch.setattr("nemo_agents_plugin.jobs.execute._MAX_LOGGED_STDERR_FILES", 2)
    for n in range(5):
        run_dir = tmp_path / f"run-{n}"
        run_dir.mkdir()
        (run_dir / "stderr.txt").write_text(f"stream {n}\n")

    with caplog.at_level(logging.WARNING, logger="nemo_agents_plugin.jobs.execute"):
        _log_agent_stderr(tmp_path)

    assert "Found 5 agent stderr files" in caplog.text
    assert "stream 0" in caplog.text
    assert "stream 1" in caplog.text
    assert "stream 4" not in caplog.text


def test_log_agent_stderr_does_not_flag_truncation_when_it_fits(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A run under the cap must read as complete, with no misleading notice."""
    (tmp_path / "stderr.txt").write_text("short and complete\n")

    with caplog.at_level(logging.ERROR, logger="nemo_agents_plugin.jobs.execute"):
        _log_agent_stderr(tmp_path)

    assert "short and complete" in caplog.text
    assert "omitted" not in caplog.text


def test_log_agent_stderr_reports_byte_counts_not_character_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap is a byte budget, so the notice has to be in bytes: multibyte
    UTF-8 would otherwise under-report by up to 4x."""
    monkeypatch.setattr("nemo_agents_plugin.jobs.execute._MAX_LOGGED_STDERR_BYTES", 400)
    # 2 bytes per character, so a character count would read as half of this.
    (tmp_path / "stderr.txt").write_text("é" * 1000, encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="nemo_agents_plugin.jobs.execute"):
        _log_agent_stderr(tmp_path)

    assert "last 400 bytes" in caplog.text
    assert "1600 earlier bytes omitted" in caplog.text


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="mkfifo is not available on this platform")
def test_log_agent_stderr_rejects_a_fifo_without_blocking(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A FIFO with no writer would block this task forever - while it is trying
    to report a failure. The artifacts dir is agent-writable, so the file type
    is not ours to trust."""
    os.mkfifo(tmp_path / "stderr.txt")

    with caplog.at_level(logging.WARNING, logger="nemo_agents_plugin.jobs.execute"):
        _log_agent_stderr(tmp_path)

    assert "is not a regular file" in caplog.text


def test_log_agent_stderr_refuses_to_follow_a_symlink(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """``st_size`` on a symlinked character device reads as 0, so it would pass
    any size cap and then read without bound (``/dev/zero`` being the obvious
    case). Refusing the symlink outright is what closes that."""
    (tmp_path / "real.txt").write_text("some diagnostics\n")
    (tmp_path / "stderr.txt").symlink_to(tmp_path / "real.txt")

    with caplog.at_level(logging.WARNING, logger="nemo_agents_plugin.jobs.execute"):
        _log_agent_stderr(tmp_path)

    assert "Could not read agent stderr" in caplog.text
    assert "some diagnostics" not in caplog.text


# ---------------------------------------------------------------------------
# Intake telemetry auto-configuration
# ---------------------------------------------------------------------------


def _fabric_agent_config(**overrides: Any) -> dict[str, Any]:
    return {
        "config_format": "nemo-agents-spec-v1",
        "name": "demo-agent",
        "default_harness": "h",
        "harnesses": {"h": {"kind": "hermes"}},
        "models": {"default": {"provider": "platform", "model": "default/m"}},
        "environment": {"provider": "local"},
        **overrides,
    }


def test_telemetry_is_pointed_at_the_workspace_intake_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the task knows the platform URL reachable from its own pod."""
    monkeypatch.setenv("NHX_BASE_URL", "http://nemo-helix-api:8080")
    config = _fabric_agent_config()

    _configure_intake_telemetry(config, workspace="team-a", proxy_origin=None)

    telemetry = config["telemetry"]
    assert telemetry["enabled"] is True
    assert telemetry["provider"] == "relay"
    assert telemetry["agent_name"] == "demo-agent"
    storage = telemetry["atif"]["storage"][0]
    assert storage["type"] == "http"
    assert storage["endpoint"] == "http://nemo-helix-api:8080/apis/intake/v2/workspaces/team-a/ingest/atif"
    # The wired config still has to be a valid agent config.
    assert AgentConfig.model_validate(config).telemetry.enabled is True


def test_telemetry_is_pointed_at_the_proxy_and_carries_no_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fabric writes the config into artifacts that are uploaded as a job result.

    The proxy authenticates the export as it forwards it, so nothing about
    identity -- inline or by environment variable name -- belongs in the config.
    """
    monkeypatch.setenv("NHX_BASE_URL", "http://nemo-helix-api:8080")
    config = _fabric_agent_config()

    _configure_intake_telemetry(config, workspace="team-a", proxy_origin="http://127.0.0.1:54321")

    storage = config["telemetry"]["atif"]["storage"][0]
    assert storage["endpoint"] == "http://127.0.0.1:54321/apis/intake/v2/workspaces/team-a/ingest/atif"
    assert "headers" not in storage, "an inline header would land in a downloadable artifact"
    assert "header_env" not in storage, "the proxy stamps identity; the config names no credential"
    assert not [name for name in os.environ if name.startswith("NHX_AGENT_TELEMETRY_HEADER_")]


@pytest.mark.parametrize(
    ("output", "declaration"),
    [
        ("atof", {"enabled": True, "sinks": [{"type": "stream", "url": "https://mine/events"}]}),
        ("opentelemetry", {"endpoints": [{"type": "gen_ai", "endpoint": "https://mine/otlp"}]}),
    ],
)
def test_another_outputs_destination_does_not_speak_for_atif(
    monkeypatch: pytest.MonkeyPatch, output: str, declaration: dict[str, Any]
) -> None:
    """Declaring a collector is an opinion about that output, not about the trajectory.

    Leaving ATIF unset is no opinion about ATIF, so the Intake default applies
    and the agent keeps the destination it did choose.
    """
    monkeypatch.setenv("NHX_BASE_URL", "http://nemo-helix-api:8080")
    config = _fabric_agent_config(telemetry={"enabled": True, output: declaration})

    _configure_intake_telemetry(config, workspace="team-a", proxy_origin=None)

    storage = config["telemetry"]["atif"]["storage"][0]
    assert storage["endpoint"] == "http://nemo-helix-api:8080/apis/intake/v2/workspaces/team-a/ingest/atif"
    assert config["telemetry"][output] == declaration


def test_an_agent_that_names_its_own_destination_keeps_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit export destination beats an inferred one."""
    monkeypatch.setenv("NHX_BASE_URL", "http://nemo-helix-api:8080")
    mine = {"type": "http", "endpoint": "https://elsewhere.example/ingest"}
    config = _fabric_agent_config(telemetry={"enabled": True, "atif": {"enabled": True, "storage": [mine]}})

    _configure_intake_telemetry(config, workspace="default", proxy_origin=None)

    assert config["telemetry"]["atif"]["storage"] == [mine]


def test_telemetry_disabled_on_the_agent_is_an_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """False means no, as distinct from a config that never mentioned telemetry."""
    monkeypatch.setenv("NHX_BASE_URL", "http://nemo-helix-api:8080")
    config = _fabric_agent_config(telemetry={"enabled": False})

    _configure_intake_telemetry(config, workspace="default", proxy_origin=None)

    assert config["telemetry"] == {"enabled": False}


def test_an_agent_that_only_names_itself_is_still_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    """The point of the tri-state: naming yourself is not configuring an export."""
    monkeypatch.setenv("NHX_BASE_URL", "http://nemo-helix-api:8080")
    config = _fabric_agent_config(telemetry={"agent_name": "my-agent-name"})

    _configure_intake_telemetry(config, workspace="default", proxy_origin=None)

    assert config["telemetry"]["enabled"] is True
    assert config["telemetry"]["agent_name"] == "my-agent-name"


def test_telemetry_is_skipped_when_no_platform_url_is_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run untraced beats a run that fails over its own tracing."""
    monkeypatch.delenv("NHX_BASE_URL", raising=False)
    config = _fabric_agent_config()

    _configure_intake_telemetry(config, workspace="default", proxy_origin=None)

    assert "telemetry" not in config


def test_an_unrecognized_telemetry_section_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deployments never model-validate, so a stray key must not fail the whole config."""
    monkeypatch.setenv("NHX_BASE_URL", "http://nemo-helix-api:8080")
    config = _fabric_agent_config(telemetry={"enabled": True, "not_a_real_field": 1})

    _configure_intake_telemetry(config, workspace="default", proxy_origin=None)

    assert config["telemetry"] == {"enabled": True, "not_a_real_field": 1}


@requires_hermes_adapter
def test_relay_support_is_read_from_the_adapter_descriptor(tmp_path: Path) -> None:
    """The bundled harnesses advertise relay with an ATIF output."""
    assert supports_intake_atif_export(_fabric_agent_config(), base_dir=tmp_path) is True


def test_an_unplannable_config_is_not_wired(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """We cannot know, so we do not wire; the invocation reports the real problem."""
    config = _fabric_agent_config(harnesses={"h": {"kind": "codex"}})

    with caplog.at_level(logging.WARNING):
        assert supports_intake_atif_export(config, base_dir=tmp_path) is False

    assert "Could not read adapter telemetry support" in caplog.text


def test_an_adapter_without_the_atif_output_is_not_wired(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relay support alone is not enough: an adapter may offer only OpenTelemetry."""
    otel_only = {"providers": {"relay": {"outputs": ["otel"]}}}

    class _Plan:
        @staticmethod
        def to_dict() -> dict[str, Any]:
            return {"adapter_descriptor": {"descriptor": {"telemetry": otel_only}}}

    monkeypatch.setattr(intake_export.fabric, "Fabric", lambda: SimpleNamespace(plan=lambda *a, **k: _Plan()))

    with caplog.at_level(logging.INFO):
        assert supports_intake_atif_export(_fabric_agent_config(), base_dir=tmp_path) is False

    assert "not its ATIF output" in caplog.text


@requires_hermes_adapter
def test_a_job_with_no_gateway_models_still_exports_through_a_proxy(
    ctx: JobContext, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Telemetry is a reason to run a proxy on its own.

    Sizing the proxy off the models alone would leave an agent that reaches no
    gateway -- this one -- posting its trajectory unauthenticated.
    """
    monkeypatch.setenv("NHX_BASE_URL", "http://nemo-helix-api:8080")
    token_file = tmp_path / "subject-token"
    token_file.write_text("subject", encoding="utf-8")
    monkeypatch.setenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", str(token_file))
    monkeypatch.setattr(
        gateway_proxy,
        "resolve_workload_exchange_provider",
        lambda **_kwargs: SimpleNamespace(get_access_token=lambda: "exchanged-token"),
    )
    agent = _resolved_agent()
    assert not gateway_proxy.routes_inference_through_gateway(AgentConfig.model_validate(agent.config))
    spec = ExecuteAgentStepConfig(request=ExecuteAgentJobConfig(agent="calc", input="hello"), agent=agent)
    seen: dict[str, Any] = {}

    async def _invoke(request: Any) -> FabricRuntimeResult:
        seen["telemetry"] = request.agent_config.telemetry.model_dump(exclude_none=True)
        return FabricRuntimeResult(status="succeeded", output={"answer": "done"})

    with patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke):
        assert ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx, sdk=MagicMock())["status"] == "completed"

    storage = seen["telemetry"]["atif"]["storage"][0]
    assert urlsplit(storage["endpoint"]).hostname == "127.0.0.1"
    assert storage["endpoint"].endswith("/apis/intake/v2/workspaces/default/ingest/atif")
    assert "header_env" not in storage


@requires_hermes_adapter
def test_an_auth_disabled_platform_exports_straight_to_the_platform(
    ctx: JobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No identity to forward means no proxy, and the export goes where it always did."""
    monkeypatch.delenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", raising=False)
    monkeypatch.delenv("NHX_PRINCIPAL", raising=False)
    monkeypatch.setenv("NHX_BASE_URL", "http://nemo-helix-api:8080")
    spec = ExecuteAgentStepConfig(request=ExecuteAgentJobConfig(agent="calc", input="hello"), agent=_resolved_agent())
    seen: dict[str, Any] = {}

    async def _invoke(request: Any) -> FabricRuntimeResult:
        seen["telemetry"] = request.agent_config.telemetry.model_dump(exclude_none=True)
        return FabricRuntimeResult(status="succeeded", output={"answer": "done"})

    with patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke):
        assert ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx, sdk=MagicMock())["status"] == "completed"

    storage = seen["telemetry"]["atif"]["storage"][0]
    assert storage["endpoint"] == "http://nemo-helix-api:8080/apis/intake/v2/workspaces/default/ingest/atif"
    assert "header_env" not in storage


def test_declining_auto_telemetry_submits_the_agent_config_as_written(
    ctx: JobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The request governs server-side filling; the agent config governs the agent.

    An agent that declares its own export still exports -- saying a run should
    not be traced is the agent config's job, and an inline agent can say it.
    Driven through ``run`` so the opt-out branch is what is under test, rather
    than the request parsing around it.
    """
    monkeypatch.setenv("NHX_BASE_URL", "http://nemo-helix-api:8080")
    # ATIF is left unset, so this config *would* be wired -- otherwise the test
    # would pass whether or not auto_telemetry was honoured.
    declared = {"enabled": True, "opentelemetry": {"endpoints": [{"type": "gen_ai", "endpoint": "https://mine"}]}}
    agent = _resolved_agent()
    agent.config["telemetry"] = declared
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello", auto_telemetry=False),
        agent=agent,
    )
    seen: dict[str, Any] = {}

    async def _invoke(request: Any) -> FabricRuntimeResult:
        seen["telemetry"] = request.agent_config.telemetry.model_dump(exclude_none=True)
        return FabricRuntimeResult(status="succeeded", output={"answer": "done"})

    with patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke):
        result = ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx, sdk=MagicMock())

    assert result["status"] == "completed"
    # Reached Fabric exactly as declared: no Intake destination added beside it.
    assert "atif" not in seen["telemetry"]
    assert seen["telemetry"]["opentelemetry"] == declared["opentelemetry"]


def test_a_telemetry_only_job_runs_untraced_rather_than_failing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Telemetry is not worth failing an agent over.

    A job with no gateway models needs the proxy only to authenticate its
    export, so a proxy that cannot start degrades to an unauthenticated post --
    what a failed token exchange produced when the export carried its own
    headers. A job whose inference depends on the proxy still fails; that is
    ``test_workload_identity_takes_precedence_without_fallback``.
    """
    monkeypatch.setenv("NHX_BASE_URL", "http://nemo-helix-api:8080")
    token_file = tmp_path / "subject-token"
    token_file.write_text("subject", encoding="utf-8")
    monkeypatch.setenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", str(token_file))

    def explode(**_kwargs: Any) -> Any:
        raise RuntimeError("auth discovery unavailable")

    monkeypatch.setattr(gateway_proxy, "resolve_workload_exchange_provider", explode)

    with caplog.at_level(logging.WARNING):
        with gateway_proxy.optional_platform_auth_proxy() as origin:
            assert origin is None

    assert "without credentials" in caplog.text


@requires_hermes_adapter
def test_a_telemetry_only_job_completes_when_the_proxy_cannot_start(
    ctx: JobContext, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The headline behavior of the optional proxy, driven through ``run``.

    ``test_a_telemetry_only_job_runs_untraced_rather_than_failing`` proves the
    context manager yields ``None`` when the proxy cannot start; this drives the
    same failure through ``run`` so a regression that made a dead proxy abort the
    run is caught directly, not only by composition. The export degrades to the
    platform origin unauthenticated -- what a failed token exchange used to
    produce -- and the run still completes. (Auth-disabled runs hit a different
    path: no identity at all yields ``None`` cleanly. This one has an identity
    whose credential discovery raises, the case the optional proxy exists for.)
    """
    monkeypatch.setenv("NHX_BASE_URL", "http://nemo-helix-api:8080")
    token_file = tmp_path / "subject-token"
    token_file.write_text("subject", encoding="utf-8")
    monkeypatch.setenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", str(token_file))

    def explode(**_kwargs: Any) -> Any:
        raise RuntimeError("auth discovery unavailable")

    monkeypatch.setattr(gateway_proxy, "resolve_workload_exchange_provider", explode)
    agent = _resolved_agent()
    assert not gateway_proxy.routes_inference_through_gateway(AgentConfig.model_validate(agent.config))
    spec = ExecuteAgentStepConfig(request=ExecuteAgentJobConfig(agent="calc", input="hello"), agent=agent)
    seen: dict[str, Any] = {}

    async def _invoke(request: Any) -> FabricRuntimeResult:
        seen["telemetry"] = request.agent_config.telemetry.model_dump(exclude_none=True)
        return FabricRuntimeResult(status="succeeded", output={"answer": "done"})

    with caplog.at_level(logging.WARNING):
        with patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke):
            result = ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx, sdk=MagicMock())

    assert result["status"] == "completed", result
    # The proxy never started: its credential discovery raised, the optional
    # wrapper degraded, and the export fell back to the platform origin instead
    # of a 127.0.0.1 loopback. The run completing is the point of the test.
    assert "without credentials" in caplog.text
    storage = seen["telemetry"]["atif"]["storage"][0]
    assert urlsplit(storage["endpoint"]).hostname == "nemo-helix-api"
    assert "header_env" not in storage


def test_an_atif_block_turned_on_without_a_destination_is_filled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turning ATIF on without a destination asks for one, rather than declaring one.

    Otherwise exporting OpenTelemetry to your own collector would cost you the
    platform trajectory, recoverable only by hand-writing the endpoint and
    header names this wiring exists to spare people.
    """
    monkeypatch.setenv("NHX_BASE_URL", "http://nemo-helix-api:8080")
    mine = {"endpoints": [{"type": "gen_ai", "endpoint": "https://mine/otlp"}]}
    config = _fabric_agent_config(telemetry={"enabled": True, "atif": {"enabled": True}, "opentelemetry": mine})

    _configure_intake_telemetry(config, workspace="team-a", proxy_origin=None)

    storage = config["telemetry"]["atif"]["storage"][0]
    assert storage["endpoint"] == "http://nemo-helix-api:8080/apis/intake/v2/workspaces/team-a/ingest/atif"
    assert config["telemetry"]["opentelemetry"] == mine, "the collector the agent chose is untouched"


# ---------------------------------------------------------------------------
# wants_intake_atif_export: the cheap half of the wiring decision
#
# Separated from supports_intake_atif_export so a caller can answer "does this
# config even want a destination?" without resolving a Fabric plan. It reads
# the same tri-state telemetry section the wiring itself does, so these pin
# that the two cannot drift apart.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("telemetry", "wanted"),
    [
        pytest.param(None, True, id="no-telemetry-section"),
        pytest.param({}, True, id="empty-telemetry-section"),
        pytest.param({"enabled": True}, True, id="enabled-without-destination"),
        pytest.param({"enabled": False}, False, id="explicit-opt-out"),
        pytest.param({"atif": {"enabled": False}}, False, id="atif-declined"),
        pytest.param(
            {"atif": {"storage": [{"type": "http", "endpoint": "https://mine/atif"}]}},
            False,
            id="destination-already-declared",
        ),
        pytest.param(
            {"opentelemetry": {"endpoints": [{"type": "gen_ai", "endpoint": "https://mine"}]}},
            True,
            id="otel-only-is-no-opinion-about-atif",
        ),
    ],
)
def test_wants_intake_atif_export_reads_the_telemetry_tri_state(telemetry: dict[str, Any] | None, wanted: bool) -> None:
    config = _fabric_agent_config()
    if telemetry is not None:
        config["telemetry"] = telemetry

    assert wants_intake_atif_export(config) is wanted


def test_wants_intake_atif_export_declines_an_unreadable_telemetry_section() -> None:
    """Same answer the wiring gives: leave a section we cannot parse alone."""
    config = _fabric_agent_config()
    config["telemetry"] = {"enabled": "yes-please"}

    assert wants_intake_atif_export(config) is False


def test_wants_intake_atif_export_agrees_with_the_wiring_it_guards() -> None:
    """The predicate must not say no to a config the wiring would have wired."""
    for telemetry in ({}, {"enabled": True}, {"enabled": False}, {"atif": {"enabled": False}}):
        config = _fabric_agent_config()
        config["telemetry"] = telemetry
        wired = intake_export.configure_intake_atif_export(config, workspace="default", base_url="http://platform:8080")
        assert wants_intake_atif_export({**config, "telemetry": telemetry}) is wired


def _agent_with_probe_skill(*, name: str = "calc", workspace_dir: str | None = None) -> ResolvedAgentConfig:
    agent = _resolved_agent(name=name)
    agent.config["skills"] = {"paths": ["skills/probe-skill"]}
    if workspace_dir is not None:
        agent.config["environment"] = {"provider": "local", "workspace": workspace_dir}
    return agent


def _write_probe_skill(local_path: Path) -> None:
    skill = local_path / "skills" / "probe-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# probe\n", encoding="utf-8")


def test_run_stages_ethos_skills_before_invoke_and_keeps_workdir(
    ctx: JobContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_real_ethos_stage(monkeypatch)
    requested: list[str] = []

    async def _download(
        _files_client: object,
        *,
        workspace: str,
        fileset_name: str,
        local_path: Path,
    ) -> None:
        assert workspace == "default"
        requested.append(fileset_name)
        _write_probe_skill(local_path)

    monkeypatch.setattr("nemo_agents_plugin.runner.fabric_artifact_staging._download_fileset", _download)

    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(
            agent="calc",
            input="hello",
            workdir=AgentWorkdir(base_workdir="source#project/"),
        ),
        agent=_agent_with_probe_skill(),
        workdir=AgentWorkdir(base_workdir="default/source#notes.txt"),
    )

    def _download_workdir(_files_client: object, ref: str, *, local_dir: Path) -> SimpleNamespace:
        del _files_client
        assert ref == "default/source#notes.txt"
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "notes.txt").write_text("from the workdir\n", encoding="utf-8")
        return SimpleNamespace(path=local_dir / "notes.txt", tmp_dir=local_dir)

    async def _invoke(request: Any) -> FabricRuntimeResult:
        assert (request.base_dir / "skills" / "probe-skill" / "SKILL.md").read_text(encoding="utf-8") == "# probe\n"
        assert (request.base_dir / "workspace" / "notes.txt").read_text(encoding="utf-8") == "from the workdir\n"
        return FabricRuntimeResult(status="succeeded", response="done")

    sdk = _sdk_opening_async_client()
    with (
        patch("nemo_agents_plugin.jobs.execute.FilesClient.from_client", return_value=sdk.files),
        patch("nemo_agents_plugin.tasks.execute.workdir._download_fileset_ref", side_effect=_download_workdir),
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
    ):
        result = ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx, sdk=sdk)

    assert result["status"] == "completed"
    assert requested == ["calc-ethos"]
    assert (ctx.storage.persistent / "results" / "input_workdir" / "notes.txt").read_text(encoding="utf-8") == (
        "from the workdir\n"
    )
    assert not (ctx.storage.persistent / "results" / "input_workdir" / "skills").exists()


@pytest.mark.parametrize("sdk", ["client", None], ids=["missing-fileset", "no-sdk"])
def test_run_rejects_declared_skills_when_ethos_cannot_be_staged(
    ctx: JobContext, monkeypatch: pytest.MonkeyPatch, sdk: str | None
) -> None:
    _use_real_ethos_stage(monkeypatch)
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_agent_with_probe_skill(),
    )

    with (
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", AsyncMock()) as invoke,
        pytest.raises(FabricArtifactStagingError, match="skills/probe-skill"),
    ):
        ExecuteAgentJob().run(
            spec.model_dump(mode="json"),
            ctx=ctx,
            sdk=None if sdk is None else _sdk_opening_async_client(),
        )

    invoke.assert_not_called()


def test_run_keeps_workdir_when_workspace_is_the_fabric_base(ctx: JobContext, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_real_ethos_stage(monkeypatch)

    async def _download(
        _files_client: object,
        *,
        workspace: str,
        fileset_name: str,
        local_path: Path,
    ) -> None:
        del workspace, fileset_name
        _write_probe_skill(local_path)

    monkeypatch.setattr("nemo_agents_plugin.runner.fabric_artifact_staging._download_fileset", _download)

    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(
            agent="calc",
            input="hello",
            workdir=AgentWorkdir(base_workdir="source#project/"),
        ),
        agent=_agent_with_probe_skill(workspace_dir="."),
        workdir=AgentWorkdir(base_workdir="default/source#notes.txt"),
    )

    def _download_workdir(_files_client: object, ref: str, *, local_dir: Path) -> SimpleNamespace:
        del _files_client, ref
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "notes.txt").write_text("from the workdir\n", encoding="utf-8")
        return SimpleNamespace(path=local_dir / "notes.txt", tmp_dir=local_dir)

    async def _invoke(request: Any) -> FabricRuntimeResult:
        assert (request.base_dir / "skills" / "probe-skill" / "SKILL.md").is_file()
        assert (request.base_dir / "notes.txt").read_text(encoding="utf-8") == "from the workdir\n"
        return FabricRuntimeResult(status="succeeded", response="done")

    sdk = _sdk_opening_async_client()
    with (
        patch("nemo_agents_plugin.jobs.execute.FilesClient.from_client", return_value=sdk.files),
        patch("nemo_agents_plugin.tasks.execute.workdir._download_fileset_ref", side_effect=_download_workdir),
        patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
    ):
        result = ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx, sdk=sdk)

    assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# Jobs stored before the ``extension`` -> ``post_run_action`` rename
# ---------------------------------------------------------------------------


def _legacy_step_spec(kind: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    extension: dict[str, Any] = {"kind": kind, "config": config or {}}
    return {
        "request": {"agent": "calc", "input": "hello", "extension": extension},
        "agent": _resolved_agent().model_dump(mode="json"),
        "extension": extension,
    }


def test_a_stored_pre_rename_step_config_reads_back_as_its_post_run_action() -> None:
    step_config = ExecuteAgentStepConfig.model_validate(
        _legacy_step_spec("insights.analysis", {"agent": "demo", "workspace": "default"})
    )

    expected = PostRunActionConfig(kind="insights.analysis", config={"agent": "demo", "workspace": "default"})
    assert step_config.post_run_action == expected
    assert step_config.request.post_run_action == expected


def test_a_stored_pre_rename_kind_reads_back_as_written() -> None:
    """``noop`` is not rewritten to ``agents.noop``: a stored kind need not be installed to be read."""
    step_config = ExecuteAgentStepConfig.model_validate(_legacy_step_spec("noop"))

    assert step_config.post_run_action == PostRunActionConfig(kind="noop")
    assert step_config.request.post_run_action == PostRunActionConfig(kind="noop")


def test_translating_a_stored_record_leaves_the_record_untouched() -> None:
    legacy = _legacy_step_spec("noop")
    snapshot = json.loads(json.dumps(legacy))

    ExecuteAgentStepConfig.model_validate(legacy)

    assert legacy == snapshot


def _read_stored_job_through_the_route(stored_spec: dict[str, Any]) -> Any:
    """GET one stored execute job through the generated job route."""
    app = FastAPI()
    app.include_router(add_job_routes(ExecuteAgentJob), prefix="/apis/agents/v2/workspaces/{workspace}")
    app.dependency_overrides[get_nemo_client] = lambda: MagicMock()

    async def _get_job(*, name: str, workspace: str) -> MagicMock:
        response = MagicMock()
        response.data.return_value = SimpleNamespace(
            id="job-1",
            attempt_id="attempt-1",
            name=name,
            description=None,
            project=None,
            workspace=workspace,
            created_at=None,
            updated_at=None,
            spec=stored_spec,
            source="nemo-agents-plugin",
            status="completed",
            status_details=None,
            error_details=None,
            ownership=None,
            custom_fields=None,
            control=None,
        )
        return response

    fake_jobs = SimpleNamespace(get_job=_get_job)
    with patch("nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client", return_value=fake_jobs):
        return TestClient(app).get("/apis/agents/v2/workspaces/default/jobs/execute/old-run")


def test_a_pre_rename_job_is_still_readable_through_the_job_route() -> None:
    """Every read renders the stored spec through ``ExecuteAgentStepConfig``; an old
    job, or the Compass analysis run backed by it, must not become a 500.
    """
    response = _read_stored_job_through_the_route(_legacy_step_spec("insights.analysis", {"agent": "demo"}))

    assert response.status_code == 200, response.text
    spec = response.json()["spec"]
    assert spec["post_run_action"] == {"kind": "insights.analysis", "config": {"agent": "demo"}}
    assert spec["request"]["post_run_action"] == spec["post_run_action"]
    assert "extension" not in spec


def test_a_job_whose_post_run_action_is_no_longer_installed_is_still_readable() -> None:
    """Only creating a job checks that its action is installed; a plugin that was
    since renamed or removed must not make its jobs unreadable.
    """
    action = {"kind": "removed-plugin.action", "config": {"anything": "goes"}}
    stored = {
        "request": {"agent": "calc", "input": "hello", "post_run_action": action},
        "agent": _resolved_agent().model_dump(mode="json"),
        "post_run_action": action,
    }

    response = _read_stored_job_through_the_route(stored)

    assert response.status_code == 200, response.text
    assert response.json()["spec"]["post_run_action"] == action


def test_a_new_request_setting_extension_is_still_rejected_by_the_create_route() -> None:
    app = FastAPI()
    app.include_router(add_job_routes(ExecuteAgentJob), prefix="/apis/agents/v2/workspaces/{workspace}")
    app.dependency_overrides[get_entity_client] = lambda: AsyncMock()
    app.dependency_overrides[get_nemo_client] = lambda: _sdk_with_files()

    response = TestClient(app, raise_server_exceptions=False).post(
        "/apis/agents/v2/workspaces/default/jobs/execute",
        json={"spec": {"agent": "calc", "input": "hello", "extension": {"kind": "agents.noop"}}},
    )

    assert response.status_code == 422
    assert "'extension' was renamed to 'post_run_action'" in response.text


@pytest.mark.parametrize("needs_skill", [False, True])
def test_run_without_optional_ethos_does_not_request_it(
    ctx: JobContext, monkeypatch: pytest.MonkeyPatch, needs_skill: bool
) -> None:
    _use_real_ethos_stage(monkeypatch)
    spec = ExecuteAgentStepConfig(
        request=ExecuteAgentJobConfig(agent="calc", input="hello"),
        agent=_agent_with_probe_skill() if needs_skill else _resolved_agent(),
        stage_ethos_fileset=False,
    )
    client = _sdk_opening_async_client()
    with (
        patch("nemo_agents_plugin.runner.fabric_artifact_staging._download_fileset", new=AsyncMock()) as download,
        patch("nemo_agents_plugin.jobs.execute.FilesClient.from_client", return_value=client.files),
        patch(
            "nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once",
            new=AsyncMock(return_value=FabricRuntimeResult(status="succeeded", response="done")),
        ) as invoke,
    ):
        if needs_skill:
            with pytest.raises(FabricArtifactStagingError):
                ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx, sdk=client)
            invoke.assert_not_awaited()
        else:
            assert ExecuteAgentJob().run(spec.model_dump(mode="json"), ctx=ctx, sdk=client)["status"] == "completed"
        download.assert_not_awaited()

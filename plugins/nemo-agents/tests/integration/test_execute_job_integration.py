# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from nemo_agents_plugin.fabric.runtime import FabricRuntimeResult
from nemo_agents_plugin.jobs.execute import ExecuteAgentJob
from nemo_agents_plugin.service import AgentsService
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.job_context import JobContext, StoragePaths
from nemo_platform_plugin.job_results import PlatformJobResults
from nmp.core.files.service import FilesService
from nmp.core.jobs.service import JobsService
from nmp.platform_runner.plugin_adapter import NemoServiceAdapter
from nmp.testing import ClientContext, create_test_client

pytestmark = pytest.mark.integration


class _TestAgentsService(NemoServiceAdapter):
    def __init__(self) -> None:
        super().__init__(AgentsService())


def _extract_tar_members(content: bytes, target_dir: Path) -> dict[str, str]:
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
        tar.extractall(target_dir, filter="data")

    files: dict[str, str] = {}
    for path in target_dir.rglob("*"):
        if path.is_file():
            relative = path.relative_to(target_dir)
            files[relative.as_posix()] = path.read_text()
    return files


def test_execute_job_materializes_layered_input_workspace(tmp_path: Path) -> None:
    with create_test_client(
        _TestAgentsService,
        FilesService,
        JobsService,
        client_type=ClientContext,
        workspace="default",
    ) as ctx:
        # Create an Agent through the public Agents API so the job's to_spec
        # resolution path sees the same entity shape as a real caller.
        agent_response = ctx.test_client.post(
            "/apis/agents/v2/workspaces/default/agents",
            json={
                "name": "calc",
                "config_format": "nemo-agents-spec-v1",
                "config": {
                    "config_format": "nemo-agents-spec-v1",
                    "name": "calc",
                    "default_harness": "hermes",
                    "harnesses": {"hermes": {"kind": "hermes"}},
                    "models": {"default": {"provider": "openai", "model": "openai/gpt-5.4"}},
                },
            },
        )
        assert agent_response.status_code == 201, agent_response.text

        ctx.sdk.files.upload_content(
            workspace="default",
            fileset="base-workdir",
            remote_path="README.md",
            content="base readme\n",
            fileset_auto_create=True,
        )
        ctx.sdk.files.upload_content(
            workspace="default",
            fileset="base-workdir",
            remote_path="app/config.yaml",
            content="source: base\n",
        )
        ctx.sdk.files.upload_content(
            workspace="default",
            fileset="config-artifact",
            remote_path="config.yaml",
            content="source: artifact\n",
            fileset_auto_create=True,
        )
        ctx.sdk.files.upload_content(
            workspace="default",
            fileset="notes-artifact",
            remote_path="notes.txt",
            content="mounted notes\n",
            fileset_auto_create=True,
        )

        job_name = "invoke-layered-workdir"
        create_response = ctx.test_client.post(
            "/apis/agents/v2/workspaces/default/jobs/execute",
            json={
                "name": job_name,
                "spec": {
                    "agent": "calc",
                    "input": "hello",
                    "workdir": {
                        "base_workdir": "base-workdir",
                        "artifact_mounts": [
                            {"ref": "config-artifact#config.yaml", "mount_path": "app/config.yaml"},
                            {"ref": "notes-artifact#notes.txt", "mount_path": "notes/notes.txt"},
                        ],
                    },
                },
            },
        )
        assert create_response.status_code == 201, create_response.text
        job = create_response.json()
        assert job["name"] == job_name
        assert job["spec"]["workdir"]["base_workdir"] == "default/base-workdir#"
        assert job["spec"]["workdir"]["artifact_mounts"] == [
            {"ref": "default/config-artifact#config.yaml", "mount_path": "app/config.yaml"},
            {"ref": "default/notes-artifact#notes.txt", "mount_path": "notes/notes.txt"},
        ]

        storage_root = tmp_path / "job-storage"
        ephemeral = storage_root / "ephemeral"
        persistent = storage_root / "persistent"
        ephemeral.mkdir(parents=True)
        persistent.mkdir(parents=True)
        job_ctx = JobContext(
            workspace="default",
            storage=StoragePaths(ephemeral=ephemeral, persistent=persistent),
            results=PlatformJobResults(
                job_name=job_name,
                workspace="default",
                client=client_from_platform(ctx.sdk, NemoClient),
            ),
            job_id=job["id"],
        )

        async def _invoke(request: Any) -> FabricRuntimeResult:
            (request.base_dir / "workspace" / "agent-output.txt").write_text("agent output\n")
            (request.base_dir / "artifacts" / "trace.json").write_text("{}\n")
            return FabricRuntimeResult(status="succeeded", response="done")

        with patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke):
            result = ExecuteAgentJob().run(job["spec"], ctx=job_ctx, sdk=ctx.sdk)
        assert result["status"] == "completed"
        assert result["input_workdir"]["name"] == "input_workdir"

        results_response = ctx.test_client.get(f"/apis/agents/v2/workspaces/default/jobs/execute/{job_name}/results")
        assert results_response.status_code == 200, results_response.text
        assert sorted(item["name"] for item in results_response.json()["data"]) == sorted(
            [
                "input_workdir",
                "fabric_run_result",
                "output_workdir",
                "output_artifacts",
            ]
        )

        result_response = ctx.test_client.get(
            f"/apis/agents/v2/workspaces/default/jobs/execute/{job_name}/results/input_workdir"
        )
        assert result_response.status_code == 200, result_response.text
        assert result_response.json()["name"] == "input_workdir"

        download_response = ctx.test_client.get(
            f"/apis/agents/v2/workspaces/default/jobs/execute/{job_name}/results/input_workdir/download"
        )
        assert download_response.status_code == 200, download_response.text
        files = _extract_tar_members(download_response.content, tmp_path / "downloaded-input-workdir")

        assert any(path.endswith("README.md") and content == "base readme\n" for path, content in files.items())
        assert any(
            path.endswith("app/config.yaml") and content == "source: artifact\n" for path, content in files.items()
        )
        assert any(path.endswith("notes/notes.txt") and content == "mounted notes\n" for path, content in files.items())
        assert "source: base\n" not in files.values()

        output_download_response = ctx.test_client.get(
            f"/apis/agents/v2/workspaces/default/jobs/execute/{job_name}/results/output_workdir/download"
        )
        assert output_download_response.status_code == 200, output_download_response.text
        output_files = _extract_tar_members(output_download_response.content, tmp_path / "downloaded-output-workdir")
        assert any(
            path.endswith("agent-output.txt") and content == "agent output\n" for path, content in output_files.items()
        )


def test_execute_job_saves_error_results_when_fabric_raises(tmp_path: Path) -> None:
    with create_test_client(
        _TestAgentsService,
        FilesService,
        JobsService,
        client_type=ClientContext,
        workspace="default",
    ) as ctx:
        agent_response = ctx.test_client.post(
            "/apis/agents/v2/workspaces/default/agents",
            json={
                "name": "calc",
                "config_format": "nemo-agents-spec-v1",
                "config": {
                    "config_format": "nemo-agents-spec-v1",
                    "name": "calc",
                    "default_harness": "hermes",
                    "harnesses": {"hermes": {"kind": "hermes"}},
                    "models": {"default": {"provider": "openai", "model": "openai/gpt-5.4"}},
                },
            },
        )
        assert agent_response.status_code == 201, agent_response.text

        ctx.sdk.files.upload_content(
            workspace="default",
            fileset="base-workdir",
            remote_path="README.md",
            content="base readme\n",
            fileset_auto_create=True,
        )

        job_name = "invoke-fabric-raises"
        create_response = ctx.test_client.post(
            "/apis/agents/v2/workspaces/default/jobs/execute",
            json={
                "name": job_name,
                "spec": {
                    "agent": "calc",
                    "input": "hello",
                    "workdir": {"base_workdir": "base-workdir"},
                },
            },
        )
        assert create_response.status_code == 201, create_response.text
        job = create_response.json()

        storage_root = tmp_path / "job-storage"
        ephemeral = storage_root / "ephemeral"
        persistent = storage_root / "persistent"
        ephemeral.mkdir(parents=True)
        persistent.mkdir(parents=True)
        job_ctx = JobContext(
            workspace="default",
            storage=StoragePaths(ephemeral=ephemeral, persistent=persistent),
            results=PlatformJobResults(
                job_name=job_name,
                workspace="default",
                client=client_from_platform(ctx.sdk, NemoClient),
            ),
            job_id=job["id"],
        )

        async def _invoke(request: Any) -> FabricRuntimeResult:
            (request.base_dir / "workspace" / "partial.txt").write_text("partial workspace\n")
            (request.base_dir / "artifacts" / "partial.log").write_text("partial artifacts\n")
            raise RuntimeError("fabric exploded")

        with (
            patch("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke),
            pytest.raises(RuntimeError, match="fabric exploded"),
        ):
            ExecuteAgentJob().run(job["spec"], ctx=job_ctx, sdk=ctx.sdk)

        results_response = ctx.test_client.get(f"/apis/agents/v2/workspaces/default/jobs/execute/{job_name}/results")
        assert results_response.status_code == 200, results_response.text
        assert sorted(item["name"] for item in results_response.json()["data"]) == sorted(
            [
                "input_workdir",
                "output_workdir",
                "output_artifacts",
                "fabric_error",
            ]
        )

        output_workdir_response = ctx.test_client.get(
            f"/apis/agents/v2/workspaces/default/jobs/execute/{job_name}/results/output_workdir/download"
        )
        assert output_workdir_response.status_code == 200, output_workdir_response.text
        output_files = _extract_tar_members(output_workdir_response.content, tmp_path / "downloaded-output-workdir")
        assert any(path.endswith("README.md") and content == "base readme\n" for path, content in output_files.items())
        assert any(
            path.endswith("partial.txt") and content == "partial workspace\n" for path, content in output_files.items()
        )

        artifacts_response = ctx.test_client.get(
            f"/apis/agents/v2/workspaces/default/jobs/execute/{job_name}/results/output_artifacts/download"
        )
        assert artifacts_response.status_code == 200, artifacts_response.text
        artifact_files = _extract_tar_members(artifacts_response.content, tmp_path / "downloaded-output-artifacts")
        assert any(
            path.endswith("partial.log") and content == "partial artifacts\n"
            for path, content in artifact_files.items()
        )

        error_response = ctx.test_client.get(
            f"/apis/agents/v2/workspaces/default/jobs/execute/{job_name}/results/fabric_error/download"
        )
        assert error_response.status_code == 200, error_response.text
        error_payload = json.loads(error_response.content)
        assert error_payload["fabric_status"] == "error"
        assert error_payload["type"] == "RuntimeError"
        assert error_payload["message"] == "fabric exploded"

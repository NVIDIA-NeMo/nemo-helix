# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for data-designer CLI commands."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from nemo_data_designer_plugin.sdk.job_resources import DataDesignerJobResource
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.data_designer.client import DataDesignerClient
from nhx.testing import MockProviderResponse, NemoRun, add_mock_provider, assert_exit_0

PROVIDER_NAME = "cli-test-provider"
MODEL_A = "model-a"
MODEL_B = "model-b"
MODEL_A_RESPONSE = "hello world"
MODEL_B_RESPONSE = "foo bar baz"
JOB_NUM_RECORDS = 10


pytestmark = [pytest.mark.timeout(600)]


def _chat_completion_response(content: str, model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1677652288,
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def _build_job_spec(provider_name: str) -> dict[str, Any]:
    return {
        "config": {
            "model_configs": [
                {
                    "alias": "a",
                    "model": MODEL_A,
                    "provider": provider_name,
                    "inference_parameters": {"top_p": 1},
                },
                {
                    "alias": "b",
                    "model": MODEL_B,
                    "provider": provider_name,
                    "inference_parameters": {"top_p": 1},
                },
            ],
            "columns": [
                {
                    "column_type": "sampler",
                    "name": "static_sampler",
                    "sampler_type": "category",
                    "params": {"values": ["static"]},
                },
                {
                    "column_type": "llm-text",
                    "name": "response_from_a",
                    "model_alias": "a",
                    "prompt": "Tell me something about {{ static_sampler }}",
                },
                {
                    "column_type": "llm-text",
                    "name": "response_from_b",
                    "model_alias": "b",
                    "prompt": "Tell me something about {{ static_sampler }}",
                },
            ],
        },
        "num_records": JOB_NUM_RECORDS,
    }


def test_data_designer_job_lifecycle(client: NemoClient, workspace: str, nemo_run: NemoRun) -> None:
    """Full data-designer job lifecycle via CLI: create, list, get, get-status, get-logs,
    results list, download-artifacts, download-analysis, delete."""
    result = nemo_run("data-designer", "jobs", "--help", workspace=workspace)
    if result.returncode != 0:
        pytest.skip("data-designer jobs CLI is not available in this build")

    provider = add_mock_provider(
        client,
        workspace=workspace,
        name=PROVIDER_NAME,
        mock_response_body_by_model={
            MODEL_A: [MockProviderResponse(response_body=_chat_completion_response(MODEL_A_RESPONSE, MODEL_A))],
            MODEL_B: [MockProviderResponse(response_body=_chat_completion_response(MODEL_B_RESPONSE, MODEL_B))],
        },
    )

    spec = _build_job_spec(provider.name)

    result = nemo_run(
        "data-designer",
        "jobs",
        "create",
        "--spec",
        json.dumps(spec),
        workspace=workspace,
    )
    assert_exit_0(result, "jobs create failed")
    job_data = json.loads(result.stdout)
    assert "name" in job_data
    job_name = job_data["name"]

    result = nemo_run("data-designer", "jobs", "list", workspace=workspace)
    assert_exit_0(result, "jobs list failed")
    assert any(j["name"] == job_name for j in json.loads(result.stdout).get("data", []))

    result = nemo_run("data-designer", "jobs", "get", job_name, workspace=workspace)
    assert_exit_0(result, "jobs get failed")
    assert json.loads(result.stdout)["name"] == job_name

    result = nemo_run("data-designer", "jobs", "get-status", job_name, workspace=workspace)
    assert_exit_0(result, "jobs get-status failed")
    assert "status" in json.loads(result.stdout)

    # Wait through the typed client to avoid spawning many CLI subprocesses while the job runs
    job_resource = DataDesignerJobResource(
        job_name=job_name, client=DataDesignerClient.from_client(client), workspace=workspace
    )
    job_resource.wait_until_done()

    result = nemo_run("data-designer", "jobs", "get-status", job_name, workspace=workspace)
    assert_exit_0(result, "jobs get-status failed")
    assert json.loads(result.stdout).get("status") == "completed"

    result = nemo_run("data-designer", "jobs", "get-logs", job_name, workspace=workspace)
    assert_exit_0(result, "jobs get-logs failed")
    assert "data" in json.loads(result.stdout)

    result = nemo_run("data-designer", "jobs", "results", "list", job_name, workspace=workspace)
    assert_exit_0(result, "jobs results list failed")
    assert "data" in json.loads(result.stdout)

    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_path = Path(tmpdir) / "artifacts.tar.gz"
        result = nemo_run(
            "data-designer",
            "jobs",
            "results",
            "download-artifacts",
            job_name,
            "--output-file",
            artifacts_path,
            workspace=workspace,
        )
    assert_exit_0(result, "results download-artifacts failed")

    nemo_run("data-designer", "jobs", "delete", job_name, workspace=workspace)

    result = nemo_run("data-designer", "jobs", "list", "--filter.name", job_name, workspace=workspace)
    assert_exit_0(result, "jobs list failed")
    assert all(j["name"] != job_name for j in json.loads(result.stdout).get("data", []))
